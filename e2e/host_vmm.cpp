#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include "acl/acl.h"

namespace {
thread_local std::string g_error;

struct SharedDescriptor {
    uint64_t magic;
    uint64_t size;
    int32_t owner_physical_device;
    int32_t owner_numa;
    std::array<uint8_t, 128> fabric_handle;
};

struct LocalMapping {
    void* imported_va = nullptr;
    aclrtDrvMemHandle imported_pa = nullptr;
    void* owner_va = nullptr;
    aclrtDrvMemHandle owner_pa = nullptr;
};

constexpr uint64_t kMagic = 0x4d37534841524544ULL;  // M7SHARED
std::unordered_map<void*, LocalMapping> g_mappings;
uint64_t g_alloc_calls = 0;
uint64_t g_free_calls = 0;

void SetError(const std::string& message) { g_error = message; }

void SetAclError(const std::string& stage, aclError ret) {
    const char* recent = aclGetRecentErrMsg();
    SetError(stage + " acl=" + std::to_string(static_cast<int>(ret)) +
             " " + (recent == nullptr ? "" : recent));
}

bool ResolveVmmMemAttr(aclrtMemAttr* mem_attr) {
    const char* value = std::getenv(
        "VLLM_ASCEND_SUPERMEM_SINGLE_VMM_GRANULARITY");
    if (value == nullptr || std::strcmp(value, "HUGE1G") == 0) {
        *mem_attr = ACL_MEM_P2P_HUGE1G;
        return true;
    }
    if (std::strcmp(value, "HUGE2M") == 0) {
        *mem_attr = ACL_MEM_P2P_HUGE;
        return true;
    }
    SetError("invalid VLLM_ASCEND_SUPERMEM_SINGLE_VMM_GRANULARITY: " +
             std::string(value));
    return false;
}

bool WaitForFile(const std::string& path, int timeout_seconds) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (access(path.c_str(), F_OK) == 0) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    SetError("timeout waiting for " + path);
    return false;
}

bool WriteAll(int fd, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    while (size != 0) {
        const ssize_t written = write(fd, bytes, size);
        if (written <= 0) return false;
        bytes += written;
        size -= static_cast<size_t>(written);
    }
    return true;
}

bool ReadAll(int fd, void* data, size_t size) {
    auto* bytes = static_cast<uint8_t*>(data);
    while (size != 0) {
        const ssize_t got = read(fd, bytes, size);
        if (got <= 0) return false;
        bytes += got;
        size -= static_cast<size_t>(got);
    }
    return true;
}

void ReleaseLocal(LocalMapping& mapping) {
    if (mapping.imported_va != nullptr) {
        (void)aclrtUnmapMem(mapping.imported_va);
        (void)aclrtReleaseMemAddress(mapping.imported_va);
        mapping.imported_va = nullptr;
    }
    if (mapping.imported_pa != nullptr) {
        (void)aclrtFreePhysical(mapping.imported_pa);
        mapping.imported_pa = nullptr;
    }
    if (mapping.owner_va != nullptr) {
        (void)aclrtUnmapMem(mapping.owner_va);
        (void)aclrtReleaseMemAddress(mapping.owner_va);
        mapping.owner_va = nullptr;
    }
    if (mapping.owner_pa != nullptr) {
        (void)aclrtFreePhysical(mapping.owner_pa);
        mapping.owner_pa = nullptr;
    }
}
}  // namespace

extern "C" {

uint64_t host_vmm_lifecycle_counts() {
    return (g_alloc_calls << 32) | g_free_calls;
}

struct HostSharedRegisteredResult {
    void* host_ptr;
    void* device_ptr;
    size_t size;
    int32_t physical_device;
    int32_t owner;
    int32_t host_observed_location_type;
    int32_t host_observed_location_id;
    int32_t device_observed_location_type;
    int32_t device_observed_location_id;
};

const char* host_shared_last_error() {
    if (!g_error.empty()) return g_error.c_str();
    const char* acl_error = aclGetRecentErrMsg();
    return acl_error == nullptr ? "" : acl_error;
}

int host_shared_registered_alloc(size_t size, int32_t logical_device,
                                 const char* path, int32_t owner,
                                 HostSharedRegisteredResult* result) {
    ++g_alloc_calls;
    if (size == 0 || path == nullptr || path[0] == '\0' || result == nullptr) {
        SetError("invalid host_shared_registered_alloc argument");
        return -1;
    }
    g_error.clear();
    *result = {};
    result->size = size;
    result->owner = owner;
    const std::string descriptor_path(path);
    const std::string ready_path = descriptor_path + ".ready";

    aclError ret = aclrtSetDevice(logical_device);
    if (ret != ACL_ERROR_NONE) {
        SetAclError("aclrtSetDevice", ret);
        return ret;
    }
    ret = aclrtGetPhyDevIdByLogicDevId(logical_device,
                                      &result->physical_device);
    if (ret != ACL_ERROR_NONE) {
        SetAclError("aclrtGetPhyDevIdByLogicDevId", ret);
        return ret;
    }

    SharedDescriptor descriptor{};
    descriptor.magic = kMagic;
    descriptor.size = size;
    LocalMapping mapping{};

    if (owner) {
        (void)unlink(ready_path.c_str());
        (void)unlink(descriptor_path.c_str());

        aclrtPhysicalMemProp prop{};
        prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        if (!ResolveVmmMemAttr(&prop.memAttr)) return -EINVAL;
        const bool allow_huge2m_fallback =
            prop.memAttr == ACL_MEM_P2P_HUGE1G;
        prop.location.type = ACL_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = result->physical_device / 2;
        ret = aclrtMallocPhysical(&mapping.owner_pa, size, &prop, 0);
        if (ret != ACL_ERROR_NONE) {
            prop.location.type = ACL_MEM_LOCATION_TYPE_HOST;
            prop.location.id = 0;
            ret = aclrtMallocPhysical(&mapping.owner_pa, size, &prop, 0);
        }
        if (ret != ACL_ERROR_NONE && allow_huge2m_fallback) {
            prop.memAttr = ACL_MEM_P2P_HUGE;
            ret = aclrtMallocPhysical(&mapping.owner_pa, size, &prop, 0);
        }
        if (ret != ACL_ERROR_NONE) {
            SetAclError("aclrtMallocPhysical(HOST)", ret);
            return ret;
        }
        descriptor.owner_physical_device = result->physical_device;
        descriptor.owner_numa = prop.location.id;
        std::fprintf(stdout, "HOST_VMM_PHYSICAL bytes=%zu mem_attr=%d location=%d:%d\n",
                     size, static_cast<int>(prop.memAttr),
                     static_cast<int>(prop.location.type), prop.location.id);
        std::fflush(stdout);

        ret = aclrtReserveMemAddress(&mapping.owner_va, size, 0, nullptr, 1);
        if (ret == ACL_ERROR_NONE) {
            ret = aclrtMapMem(mapping.owner_va, size, 0, mapping.owner_pa, 0);
        }
        if (ret != ACL_ERROR_NONE) {
            SetAclError("owner reserve/map", ret);
            ReleaseLocal(mapping);
            return ret;
        }

        aclrtMemFabricHandle handle{};
        ret = aclrtMemExportToShareableHandleV2(
            mapping.owner_pa,
            ACL_RT_VMM_EXPORT_FLAG_DISABLE_PID_VALIDATION,
            ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, &handle);
        if (ret != ACL_ERROR_NONE) {
            SetAclError("aclrtMemExportToShareableHandleV2", ret);
            ReleaseLocal(mapping);
            return ret;
        }
        std::memcpy(descriptor.fabric_handle.data(), handle.data,
                    descriptor.fabric_handle.size());

        const int fd = open(descriptor_path.c_str(),
                            O_CREAT | O_EXCL | O_WRONLY, 0600);
        if (fd < 0 || !WriteAll(fd, &descriptor, sizeof(descriptor))) {
            if (fd >= 0) close(fd);
            SetError("descriptor write failed: " +
                     std::string(strerror(errno)));
            ReleaseLocal(mapping);
            return -errno;
        }
        close(fd);
    } else {
        if (!WaitForFile(ready_path, 120)) return -ETIMEDOUT;
        const int fd = open(descriptor_path.c_str(), O_RDONLY);
        if (fd < 0 || !ReadAll(fd, &descriptor, sizeof(descriptor))) {
            if (fd >= 0) close(fd);
            SetError("descriptor read failed: " +
                     std::string(strerror(errno)));
            return -errno;
        }
        close(fd);
        if (descriptor.magic != kMagic || descriptor.size != size) {
            SetError("shared VMM descriptor mismatch");
            return -EINVAL;
        }
    }

    aclrtMemFabricHandle imported_handle{};
    std::memcpy(imported_handle.data, descriptor.fabric_handle.data(),
                descriptor.fabric_handle.size());
    ret = aclrtMemImportFromShareableHandleV2(
        &imported_handle, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0,
        &mapping.imported_pa);
    if (ret == ACL_ERROR_NONE) {
        ret = aclrtReserveMemAddress(&mapping.imported_va, size, 0, nullptr, 1);
    }
    if (ret == ACL_ERROR_NONE) {
        ret = aclrtMapMem(mapping.imported_va, size, 0, mapping.imported_pa, 0);
    }
    if (ret != ACL_ERROR_NONE) {
        SetAclError("self/cross import reserve/map", ret);
        ReleaseLocal(mapping);
        return ret;
    }

    if (owner) {
        const int marker = open(ready_path.c_str(),
                                O_CREAT | O_EXCL | O_WRONLY, 0600);
        if (marker < 0) {
            SetError("ready marker create failed: " +
                     std::string(strerror(errno)));
            ReleaseLocal(mapping);
            return -errno;
        }
        close(marker);
    }

    result->host_ptr = owner ? mapping.owner_va : mapping.imported_va;
    result->device_ptr = mapping.imported_va;
    aclrtPtrAttributes attributes{};
    if (owner &&
        aclrtPointerGetAttributes(mapping.owner_va, &attributes) ==
            ACL_ERROR_NONE) {
        result->host_observed_location_type = attributes.location.type;
        result->host_observed_location_id = attributes.location.id;
    } else {
        result->host_observed_location_type = ACL_MEM_LOCATION_TYPE_HOST_NUMA;
        result->host_observed_location_id = descriptor.owner_numa;
    }
    attributes = {};
    if (aclrtPointerGetAttributes(mapping.imported_va, &attributes) ==
        ACL_ERROR_NONE) {
        result->device_observed_location_type = attributes.location.type;
        result->device_observed_location_id = attributes.location.id;
    }
    g_mappings.emplace(result->device_ptr, mapping);
    return ACL_ERROR_NONE;
}

int host_shared_registered_free(HostSharedRegisteredResult* result) {
    ++g_free_calls;
    if (result == nullptr || result->device_ptr == nullptr ||
        result->size == 0) {
        SetError("invalid host_shared_registered_free argument");
        return -1;
    }
    auto found = g_mappings.find(result->device_ptr);
    if (found == g_mappings.end()) {
        SetError("shared VMM mapping not found");
        return -ENOENT;
    }
    ReleaseLocal(found->second);
    g_mappings.erase(found);
    result->host_ptr = nullptr;
    result->device_ptr = nullptr;
    return ACL_ERROR_NONE;
}

int host_shared_registered_unlink(const char* path) {
    if (path == nullptr || path[0] == '\0') return -1;
    const std::string descriptor_path(path);
    const std::string ready_path = descriptor_path + ".ready";
    (void)unlink(ready_path.c_str());
    if (unlink(descriptor_path.c_str()) != 0 && errno != ENOENT) return -errno;
    return 0;
}

}  // extern "C"
