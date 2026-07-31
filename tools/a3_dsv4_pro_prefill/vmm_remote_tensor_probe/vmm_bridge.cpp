// Copyright (c) 2026.
// SPDX-License-Identifier: Apache-2.0
//
// A deliberately small C lifecycle wrapper for the DSA-CP Ascend VMM gate.
// Python receives only opaque region handles, mapped pointers, sizes, and the
// 128-byte V2 shareable handle. All ACL object ownership stays in this file.

#include <acl/acl_rt.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <sstream>
#include <string>

namespace {

struct VmmRegion {
  int32_t device_id = -1;
  void* virtual_address = nullptr;
  size_t mapped_size = 0;
  aclrtDrvMemHandle physical_handle = nullptr;
  bool mapped = false;
  bool owns_physical_handle = false;
};

thread_local std::string last_error;
thread_local uint64_t physical_allocation_call_count = 0;
thread_local uint64_t import_call_count = 0;

int Fail(const char* api, aclError status) {
  std::ostringstream message;
  message << api << " failed with aclError=" << status;
  last_error = message.str();
  return static_cast<int>(status == ACL_SUCCESS ? -1 : status);
}

int FailMessage(const char* message) {
  last_error = message;
  return -1;
}

aclrtPhysicalMemProp PhysicalMemoryProperties(int32_t device_id) {
  aclrtPhysicalMemProp properties = {};
  properties.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  properties.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  // Match the raw G27 V2 PASS exactly; its minimum allocation granularity on
  // this .204 runtime is 2 MiB.
  properties.memAttr = ACL_HBM_MEM_NORMAL;
  properties.location.id = static_cast<uint32_t>(device_id);
  properties.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  properties.reserve = 0;
  return properties;
}

size_t AlignUp(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

int SetDevice(int32_t device_id) {
  const aclError status = aclrtSetDevice(device_id);
  return status == ACL_SUCCESS ? 0 : Fail("aclrtSetDevice", status);
}

int DestroyRegion(VmmRegion* region) {
  if (region == nullptr) {
    return 0;
  }

  int first_error = 0;
  if (SetDevice(region->device_id) != 0) {
    first_error = -1;
  }
  if (region->mapped) {
    const aclError status = aclrtUnmapMem(region->virtual_address);
    if (status != ACL_SUCCESS && first_error == 0) {
      first_error = Fail("aclrtUnmapMem", status);
    }
    region->mapped = false;
  }
  if (region->owns_physical_handle && region->physical_handle != nullptr) {
    const aclError status = aclrtFreePhysical(region->physical_handle);
    if (status != ACL_SUCCESS && first_error == 0) {
      first_error = Fail("aclrtFreePhysical", status);
    }
    region->physical_handle = nullptr;
    region->owns_physical_handle = false;
  }
  if (region->virtual_address != nullptr) {
    const aclError status = aclrtReleaseMemAddress(region->virtual_address);
    if (status != ACL_SUCCESS && first_error == 0) {
      first_error = Fail("aclrtReleaseMemAddress", status);
    }
    region->virtual_address = nullptr;
  }
  delete region;
  return first_error;
}

}  // namespace

extern "C" {

const char* dsa_vmm_last_error() {
  return last_error.c_str();
}

size_t dsa_vmm_v2_handle_size() {
  return sizeof(aclrtMemFabricHandle);
}

int dsa_vmm_get_bare_tgid(int32_t device_id, int32_t* bare_tgid) {
  if (bare_tgid == nullptr) {
    return FailMessage("dsa_vmm_get_bare_tgid: bare_tgid is null");
  }
  if (SetDevice(device_id) != 0) {
    return -1;
  }
  const aclError status = aclrtDeviceGetBareTgid(bare_tgid);
  return status == ACL_SUCCESS ? 0 : Fail("aclrtDeviceGetBareTgid", status);
}

int dsa_vmm_enable_peer(int32_t device_id, int32_t peer_device_id,
                        int32_t* can_access_peer) {
  if (can_access_peer == nullptr) {
    return FailMessage("dsa_vmm_enable_peer: can_access_peer is null");
  }
  if (SetDevice(device_id) != 0) {
    return -1;
  }
  aclError status =
      aclrtDeviceCanAccessPeer(can_access_peer, device_id, peer_device_id);
  if (status != ACL_SUCCESS) {
    return Fail("aclrtDeviceCanAccessPeer", status);
  }
  if (*can_access_peer != 1) {
    return FailMessage("aclrtDeviceCanAccessPeer returned can_access_peer != 1");
  }
  status = aclrtDeviceEnablePeerAccess(peer_device_id, 0);
  return status == ACL_SUCCESS ? 0
                               : Fail("aclrtDeviceEnablePeerAccess", status);
}

int dsa_vmm_hbm_mem_info(int32_t device_id, size_t* free_bytes,
                         size_t* total_bytes) {
  if (free_bytes == nullptr || total_bytes == nullptr) {
    return FailMessage("dsa_vmm_hbm_mem_info: output is null");
  }
  if (SetDevice(device_id) != 0) {
    return -1;
  }
  const aclError status =
      aclrtGetMemInfo(ACL_HBM_MEM, free_bytes, total_bytes);
  return status == ACL_SUCCESS ? 0 : Fail("aclrtGetMemInfo", status);
}

int dsa_vmm_get_call_counts(uint64_t* physical_allocation_calls,
                            uint64_t* import_calls) {
  if (physical_allocation_calls == nullptr || import_calls == nullptr) {
    return FailMessage("dsa_vmm_get_call_counts: output is null");
  }
  *physical_allocation_calls = physical_allocation_call_count;
  *import_calls = import_call_count;
  return 0;
}

int dsa_vmm_create_local(int32_t device_id, size_t requested_size,
                         void** opaque_region) {
  if (opaque_region == nullptr || requested_size == 0) {
    return FailMessage(
        "dsa_vmm_create_local: output is null or requested_size is zero");
  }
  *opaque_region = nullptr;
  if (SetDevice(device_id) != 0) {
    return -1;
  }

  VmmRegion* region = new (std::nothrow) VmmRegion();
  if (region == nullptr) {
    return FailMessage("dsa_vmm_create_local: region allocation failed");
  }
  region->device_id = device_id;

  aclrtPhysicalMemProp properties = PhysicalMemoryProperties(device_id);
  size_t granularity = 0;
  aclError status = aclrtMemGetAllocationGranularity(
      &properties, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity);
  if (status != ACL_SUCCESS) {
    delete region;
    return Fail("aclrtMemGetAllocationGranularity", status);
  }
  region->mapped_size = AlignUp(requested_size, granularity);

  status = aclrtReserveMemAddress(&region->virtual_address,
                                  region->mapped_size, 0, nullptr, 0);
  if (status != ACL_SUCCESS) {
    delete region;
    return Fail("aclrtReserveMemAddress", status);
  }

  ++physical_allocation_call_count;
  status = aclrtMallocPhysical(&region->physical_handle, region->mapped_size,
                               &properties, 0);
  if (status != ACL_SUCCESS) {
    const int result = Fail("aclrtMallocPhysical", status);
    DestroyRegion(region);
    return result;
  }
  region->owns_physical_handle = true;

  status = aclrtMapMem(region->virtual_address, region->mapped_size, 0,
                       region->physical_handle, 0);
  if (status != ACL_SUCCESS) {
    const int result = Fail("aclrtMapMem", status);
    DestroyRegion(region);
    return result;
  }
  region->mapped = true;
  *opaque_region = region;
  return 0;
}

int dsa_vmm_export_v2(void* opaque_region, void* output_handle,
                      size_t output_handle_size) {
  auto* region = static_cast<VmmRegion*>(opaque_region);
  if (region == nullptr || output_handle == nullptr ||
      output_handle_size != sizeof(aclrtMemFabricHandle)) {
    return FailMessage("dsa_vmm_export_v2: invalid region or handle buffer");
  }
  aclrtMemFabricHandle shareable_handle = {};
  const aclError status = aclrtMemExportToShareableHandleV2(
      region->physical_handle, ACL_RT_VMM_EXPORT_FLAG_DEFAULT,
      ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, &shareable_handle);
  if (status != ACL_SUCCESS) {
    return Fail("aclrtMemExportToShareableHandleV2", status);
  }
  std::memcpy(output_handle, &shareable_handle, sizeof(shareable_handle));
  return 0;
}

int dsa_vmm_authorize_v2(const void* shareable_handle,
                         size_t shareable_handle_size, int32_t bare_tgid) {
  if (shareable_handle == nullptr ||
      shareable_handle_size != sizeof(aclrtMemFabricHandle)) {
    return FailMessage("dsa_vmm_authorize_v2: invalid handle buffer");
  }
  aclrtMemFabricHandle handle_copy = {};
  std::memcpy(&handle_copy, shareable_handle, sizeof(handle_copy));
  int32_t trusted_process = bare_tgid;
  const aclError status = aclrtMemSetPidToShareableHandleV2(
      &handle_copy, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, &trusted_process, 1);
  return status == ACL_SUCCESS
             ? 0
             : Fail("aclrtMemSetPidToShareableHandleV2", status);
}

int dsa_vmm_import_v2(int32_t device_id, const void* shareable_handle,
                      size_t shareable_handle_size, size_t mapped_size,
                      void** opaque_region) {
  if (opaque_region == nullptr || shareable_handle == nullptr ||
      shareable_handle_size != sizeof(aclrtMemFabricHandle) ||
      mapped_size == 0) {
    return FailMessage("dsa_vmm_import_v2: invalid input");
  }
  *opaque_region = nullptr;
  if (SetDevice(device_id) != 0) {
    return -1;
  }

  VmmRegion* region = new (std::nothrow) VmmRegion();
  if (region == nullptr) {
    return FailMessage("dsa_vmm_import_v2: region allocation failed");
  }
  region->device_id = device_id;
  region->mapped_size = mapped_size;

  aclrtMemFabricHandle handle_copy = {};
  std::memcpy(&handle_copy, shareable_handle, sizeof(handle_copy));
  ++import_call_count;
  aclError status = aclrtMemImportFromShareableHandleV2(
      &handle_copy, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, 0,
      &region->physical_handle);
  if (status != ACL_SUCCESS) {
    delete region;
    return Fail("aclrtMemImportFromShareableHandleV2", status);
  }
  region->owns_physical_handle = true;

  status = aclrtReserveMemAddress(&region->virtual_address,
                                  region->mapped_size, 0, nullptr, 0);
  if (status != ACL_SUCCESS) {
    const int result = Fail("aclrtReserveMemAddress", status);
    DestroyRegion(region);
    return result;
  }
  status = aclrtMapMem(region->virtual_address, region->mapped_size, 0,
                       region->physical_handle, 0);
  if (status != ACL_SUCCESS) {
    const int result = Fail("aclrtMapMem", status);
    DestroyRegion(region);
    return result;
  }
  region->mapped = true;
  *opaque_region = region;
  return 0;
}

int dsa_vmm_set_local_access(void* opaque_region) {
  auto* region = static_cast<VmmRegion*>(opaque_region);
  if (region == nullptr) {
    return FailMessage("dsa_vmm_set_local_access: region is null");
  }
  aclrtMemAccessDesc descriptor = {};
  descriptor.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
  descriptor.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  descriptor.location.id = static_cast<uint32_t>(region->device_id);
  const aclError status = aclrtMemSetAccess(
      region->virtual_address, region->mapped_size, &descriptor, 1);
  return status == ACL_SUCCESS ? 0 : Fail("aclrtMemSetAccess", status);
}

uintptr_t dsa_vmm_region_pointer(void* opaque_region) {
  const auto* region = static_cast<const VmmRegion*>(opaque_region);
  return region == nullptr
             ? 0
             : reinterpret_cast<uintptr_t>(region->virtual_address);
}

size_t dsa_vmm_region_size(void* opaque_region) {
  const auto* region = static_cast<const VmmRegion*>(opaque_region);
  return region == nullptr ? 0 : region->mapped_size;
}

int dsa_vmm_destroy_region(void* opaque_region) {
  return DestroyRegion(static_cast<VmmRegion*>(opaque_region));
}

}  // extern "C"
