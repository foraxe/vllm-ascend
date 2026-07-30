// Two-rank A3 VMM gate for sparse owner-backed pages.
//
// Each rank owns two of four logical pages:
//   owner(logical_page) = logical_page % 2
//   owner_page          = logical_page / 2
// It allocates physical HBM only for those pages, maps them into a four-page
// logical VA reservation, then imports the peer's V2 handles into the holes.
// Blocking ACL copies plus socket barriers prove peer-read and remote-write
// aliasing. This is an allocator/transport gate, not a tensor or kernel gate.

#include <acl/acl.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0xC128A327U;
constexpr uint32_t kVersion = 1;
constexpr int kWorldSize = 2;
constexpr int kLogicalPages = 4;
constexpr int kOwnedPages = kLogicalPages / kWorldSize;
constexpr int kSocketTimeoutSeconds = 10;
constexpr int kCapabilityUnsupported = 207000;

static_assert(sizeof(aclrtMemFabricHandle) == 128,
              "CANN 9 V2 fabric handle must be 128 bytes");

enum class ResultCode : int {
  kPass = 0,
  kFailApi = 2,
  kBlockedCapability = 3,
  kFailData = 4,
  kFailCleanup = 5,
  kFailControl = 6,
};

struct PeerInfo {
  uint32_t magic;
  uint32_t version;
  int32_t rank;
  int32_t tgid;
  int32_t device;
  int32_t reserved;
  uint64_t granularity;
};

struct SharedPage {
  int32_t logical_page;
  int32_t reserved;
  aclrtMemFabricHandle handle;
};

struct HandleBundle {
  uint32_t magic;
  uint32_t version;
  int32_t rank;
  int32_t count;
  uint64_t granularity;
  SharedPage pages[kOwnedPages];
};

struct PageState {
  int logical_page = -1;
  aclrtDrvMemHandle handle = nullptr;
  void* va = nullptr;
  bool mapped = false;
};

struct ProbeState {
  int rank = -1;
  int own_device = -1;
  int peer_device = -1;
  int control = -1;
  int listener = -1;
  bool acl_initialized = false;
  bool device_set = false;
  bool peer_enabled = false;
  bool va_reserved = false;
  bool imports_released_barrier = false;
  void* base = nullptr;
  size_t granularity = 0;
  size_t free_before = 0;
  size_t free_owned = 0;
  size_t free_peak = 0;
  size_t free_after = 0;
  size_t total_hbm = 0;
  size_t local_mismatches = 0;
  size_t peer_read_mismatches = 0;
  size_t remote_write_mismatches = 0;
  std::array<PageState, kOwnedPages> owned;
  std::array<PageState, kOwnedPages> imported;
};

const char* result_name(ResultCode result) {
  switch (result) {
    case ResultCode::kPass:
      return "PASS";
    case ResultCode::kBlockedCapability:
      return "BLOCKED_CAPABILITY";
    case ResultCode::kFailData:
      return "FAIL_DATA";
    case ResultCode::kFailCleanup:
      return "FAIL_CLEANUP";
    case ResultCode::kFailControl:
      return "FAIL_CONTROL";
    default:
      return "FAIL_API";
  }
}

bool check_acl(aclError error, const char* operation) {
  if (error == ACL_SUCCESS) {
    return true;
  }
  const char* recent = aclGetRecentErrMsg();
  std::fprintf(stderr, "FAIL %s aclError=%d recent=%s\n", operation, error,
               recent == nullptr ? "<none>" : recent);
  std::fflush(stderr);
  return false;
}

bool read_full(int fd, void* buffer, size_t bytes) {
  auto* cursor = static_cast<uint8_t*>(buffer);
  while (bytes > 0) {
    const ssize_t count = read(fd, cursor, bytes);
    if (count <= 0) {
      std::fprintf(stderr, "FAIL control read errno=%d (%s)\n", errno,
                   std::strerror(errno));
      return false;
    }
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
  return true;
}

bool write_full(int fd, const void* buffer, size_t bytes) {
  const auto* cursor = static_cast<const uint8_t*>(buffer);
  while (bytes > 0) {
    const ssize_t count = write(fd, cursor, bytes);
    if (count <= 0) {
      std::fprintf(stderr, "FAIL control write errno=%d (%s)\n", errno,
                   std::strerror(errno));
      return false;
    }
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
  return true;
}

template <typename T>
bool exchange_value(int fd, int rank, const T& local, T* peer) {
  if (rank == 0) {
    return write_full(fd, &local, sizeof(local)) &&
           read_full(fd, peer, sizeof(*peer));
  }
  return read_full(fd, peer, sizeof(*peer)) &&
         write_full(fd, &local, sizeof(local));
}

bool exchange_status(int fd, int rank, int32_t local_status,
                     int32_t* peer_status, const char* phase) {
  std::fprintf(stderr, "STEP phase=%s local_status=%d\n", phase, local_status);
  std::fflush(stderr);
  return exchange_value(fd, rank, local_status, peer_status);
}

void set_socket_timeout(int fd) {
  timeval timeout{};
  timeout.tv_sec = kSocketTimeoutSeconds;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

int make_listener(const char* path) {
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::snprintf(address.sun_path, sizeof(address.sun_path), "%s", path);
  unlink(path);
  if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
      listen(fd, 1) != 0) {
    std::fprintf(stderr, "FAIL bind/listen errno=%d (%s)\n", errno,
                 std::strerror(errno));
    close(fd);
    return -1;
  }
  return fd;
}

int connect_socket(const char* path) {
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::snprintf(address.sun_path, sizeof(address.sun_path), "%s", path);
  for (int retry = 0; retry < 100; ++retry) {
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
        0) {
      set_socket_timeout(fd);
      return fd;
    }
    usleep(100000);
  }
  std::fprintf(stderr, "FAIL connect errno=%d (%s)\n", errno,
               std::strerror(errno));
  close(fd);
  return -1;
}

void* page_va(void* base, size_t granularity, int logical_page) {
  return static_cast<void*>(static_cast<uint8_t*>(base) +
                            granularity * logical_page);
}

aclrtPhysicalMemProp physical_prop(int device) {
  aclrtPhysicalMemProp prop{};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_NORMAL;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<uint32_t>(device);
  return prop;
}

uint32_t pattern_word(int logical_page, int epoch, int writer_rank,
                      size_t index) {
  return 0xC128A500U ^
         static_cast<uint32_t>(logical_page) * 0x9E3779B9U ^
         static_cast<uint32_t>(epoch) * 0x85EBCA6BU ^
         static_cast<uint32_t>(writer_rank) * 0xC2B2AE35U ^
         static_cast<uint32_t>(index) * 0x27D4EB2DU;
}

bool write_pattern(void* va, size_t bytes, int logical_page, int epoch,
                   int writer_rank) {
  std::vector<uint32_t> values(bytes / sizeof(uint32_t));
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = pattern_word(logical_page, epoch, writer_rank, index);
  }
  return check_acl(
      aclrtMemcpy(va, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
      "aclrtMemcpy pattern H2D");
}

bool validate_pattern(void* va, size_t bytes, int logical_page, int epoch,
                      int writer_rank, size_t* mismatches) {
  std::vector<uint32_t> values(bytes / sizeof(uint32_t));
  if (!check_acl(
          aclrtMemcpy(values.data(), bytes, va, bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "aclrtMemcpy pattern D2H")) {
    return false;
  }
  size_t mismatch_count = 0;
  for (size_t index = 0; index < values.size(); ++index) {
    const uint32_t expected =
        pattern_word(logical_page, epoch, writer_rank, index);
    if (values[index] != expected) {
      if (mismatch_count == 0) {
        std::fprintf(stderr,
                     "FAIL pattern page=%d epoch=%d writer=%d index=%zu "
                     "got=0x%08x expected=0x%08x\n",
                     logical_page, epoch, writer_rank, index, values[index],
                     expected);
      }
      ++mismatch_count;
    }
  }
  *mismatches += mismatch_count;
  return mismatch_count == 0;
}

void query_memory(size_t* free_bytes, size_t* total_bytes,
                  const char* label) {
  const aclError error =
      aclrtGetMemInfo(ACL_HBM_MEM, free_bytes, total_bytes);
  if (error != ACL_SUCCESS) {
    std::fprintf(stderr, "WARN %s aclrtGetMemInfo aclError=%d\n", label,
                 error);
    *free_bytes = 0;
    *total_bytes = 0;
  }
}

bool setup_control(ProbeState* state, const char* socket_path) {
  if (state->rank == 0) {
    state->listener = make_listener(socket_path);
    if (state->listener < 0) {
      return false;
    }
    state->control = accept(state->listener, nullptr, nullptr);
    if (state->control < 0) {
      return false;
    }
    set_socket_timeout(state->control);
    return true;
  }
  state->control = connect_socket(socket_path);
  return state->control >= 0;
}

ResultCode cleanup(ProbeState* state, ResultCode primary,
                   const char* socket_path) {
  bool cleanup_ok = true;

  for (auto& page : state->imported) {
    if (page.mapped) {
      cleanup_ok &=
          check_acl(aclrtUnmapMem(page.va), "aclrtUnmapMem imported");
      page.mapped = false;
    }
    if (page.handle != nullptr) {
      cleanup_ok &=
          check_acl(aclrtFreePhysical(page.handle),
                    "aclrtFreePhysical imported");
      page.handle = nullptr;
    }
  }

  if (state->control >= 0) {
    int32_t peer_status = 1;
    const int32_t local_status = cleanup_ok ? 0 : 1;
    state->imports_released_barrier =
        exchange_status(state->control, state->rank, local_status, &peer_status,
                        "IMPORTS_RELEASED") &&
        peer_status == 0;
    cleanup_ok &= state->imports_released_barrier;
  }

  // An owner allocation is freed only after the peer acknowledged that every
  // imported alias was unmapped and released.
  if (state->imports_released_barrier) {
    for (auto& page : state->owned) {
      if (page.mapped) {
        cleanup_ok &=
            check_acl(aclrtUnmapMem(page.va), "aclrtUnmapMem owned");
        page.mapped = false;
      }
    }
    if (state->va_reserved) {
      cleanup_ok &=
          check_acl(aclrtReleaseMemAddress(state->base),
                    "aclrtReleaseMemAddress");
      state->va_reserved = false;
    }
    for (auto& page : state->owned) {
      if (page.handle != nullptr) {
        cleanup_ok &=
            check_acl(aclrtFreePhysical(page.handle),
                      "aclrtFreePhysical owned");
        page.handle = nullptr;
      }
    }
    query_memory(&state->free_after, &state->total_hbm, "after");
  } else {
    std::fprintf(stderr,
                 "WARN peer did not release imports; owner handles left to "
                 "process teardown\n");
  }

  if (state->peer_enabled) {
    cleanup_ok &=
        check_acl(aclrtDeviceDisablePeerAccess(state->peer_device),
                  "aclrtDeviceDisablePeerAccess");
    state->peer_enabled = false;
  }
  if (state->device_set) {
    cleanup_ok &=
        check_acl(aclrtResetDevice(state->own_device), "aclrtResetDevice");
    state->device_set = false;
  }
  if (state->acl_initialized) {
    cleanup_ok &= check_acl(aclFinalize(), "aclFinalize");
    state->acl_initialized = false;
  }
  if (state->control >= 0) {
    close(state->control);
  }
  if (state->listener >= 0) {
    close(state->listener);
  }
  if (state->rank == 0) {
    unlink(socket_path);
  }
  if (primary == ResultCode::kPass && !cleanup_ok) {
    return ResultCode::kFailCleanup;
  }
  return primary;
}

void print_json(const ProbeState& state, ResultCode result) {
  const uint64_t logical_va_bytes =
      static_cast<uint64_t>(state.granularity) * kLogicalPages;
  const uint64_t owned_bytes =
      static_cast<uint64_t>(state.granularity) * kOwnedPages;
  std::printf(
      "{\"status\":\"%s\",\"rank\":%d,\"device\":%d,"
      "\"peer_device\":%d,\"logical_pages\":%d,"
      "\"granularity_bytes\":%zu,\"logical_va_bytes\":%llu,"
      "\"owned_physical_allocations\":%d,"
      "\"owned_physical_bytes\":%llu,\"imported_aliases\":%d,"
      "\"imported_alias_bytes\":%llu,\"free_hbm_before\":%zu,"
      "\"free_hbm_owned\":%zu,\"free_hbm_peak\":%zu,"
      "\"free_hbm_after\":%zu,\"total_hbm\":%zu,"
      "\"local_mismatches\":%zu,\"peer_read_mismatches\":%zu,"
      "\"remote_write_mismatches\":%zu,"
      "\"imports_released_barrier\":%s}\n",
      result_name(result), state.rank, state.own_device, state.peer_device,
      kLogicalPages, state.granularity,
      static_cast<unsigned long long>(logical_va_bytes), kOwnedPages,
      static_cast<unsigned long long>(owned_bytes), kOwnedPages,
      static_cast<unsigned long long>(owned_bytes), state.free_before,
      state.free_owned, state.free_peak, state.free_after, state.total_hbm,
      state.local_mismatches, state.peer_read_mismatches,
      state.remote_write_mismatches,
      state.imports_released_barrier ? "true" : "false");
  std::fflush(stdout);
}

ResultCode run_probe(ProbeState* state, const char* socket_path,
                     size_t per_rank_budget) {
  std::fprintf(stderr, "STEP aclInit rank=%d device=%d\n", state->rank,
               state->own_device);
  aclError error = aclInit(nullptr);
  if (!check_acl(error, "aclInit")) {
    return ResultCode::kFailApi;
  }
  state->acl_initialized = true;
  if (!check_acl(aclrtSetDevice(state->own_device), "aclrtSetDevice")) {
    return ResultCode::kFailApi;
  }
  state->device_set = true;

  int32_t can_access = 0;
  error = aclrtDeviceCanAccessPeer(&can_access, state->own_device,
                                   state->peer_device);
  if (error == kCapabilityUnsupported) {
    std::fprintf(stderr, "BLOCKED peer-query unsupported aclError=%d\n",
                 error);
    return ResultCode::kBlockedCapability;
  }
  if (!check_acl(error, "aclrtDeviceCanAccessPeer")) {
    return ResultCode::kFailApi;
  }
  if (can_access == 0) {
    std::fprintf(stderr, "BLOCKED peer-query can_access=0\n");
    return ResultCode::kBlockedCapability;
  }
  error = aclrtDeviceEnablePeerAccess(state->peer_device, 0);
  if (error == kCapabilityUnsupported) {
    return ResultCode::kBlockedCapability;
  }
  if (!check_acl(error, "aclrtDeviceEnablePeerAccess")) {
    return ResultCode::kFailApi;
  }
  state->peer_enabled = true;

  aclrtPhysicalMemProp prop = physical_prop(state->own_device);
  if (!check_acl(
          aclrtMemGetAllocationGranularity(
              &prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM,
              &state->granularity),
          "aclrtMemGetAllocationGranularity")) {
    return ResultCode::kFailApi;
  }
  if (state->granularity == 0 ||
      state->granularity % sizeof(uint32_t) != 0 ||
      state->granularity * kOwnedPages > per_rank_budget) {
    std::fprintf(stderr,
                 "BLOCKED granularity=%zu owned_bytes=%zu budget=%zu\n",
                 state->granularity, state->granularity * kOwnedPages,
                 per_rank_budget);
    return ResultCode::kBlockedCapability;
  }

  query_memory(&state->free_before, &state->total_hbm, "before");
  if (!check_acl(aclrtReserveMemAddress(
                     &state->base, state->granularity * kLogicalPages, 0,
                     nullptr, 0),
                 "aclrtReserveMemAddress")) {
    return ResultCode::kFailApi;
  }
  state->va_reserved = true;

  bool local_ok = true;
  for (int owner_page = 0; owner_page < kOwnedPages; ++owner_page) {
    PageState& page = state->owned[owner_page];
    page.logical_page = state->rank + owner_page * kWorldSize;
    page.va = page_va(state->base, state->granularity, page.logical_page);
    local_ok &=
        check_acl(aclrtMallocPhysical(&page.handle, state->granularity, &prop,
                                      0),
                  "aclrtMallocPhysical owned");
    if (!local_ok) {
      break;
    }
    local_ok &=
        check_acl(aclrtMapMem(page.va, state->granularity, 0, page.handle, 0),
                  "aclrtMapMem owned");
    page.mapped = local_ok;
    local_ok &=
        write_pattern(page.va, state->granularity, page.logical_page, 0,
                      state->rank);
    local_ok &=
        validate_pattern(page.va, state->granularity, page.logical_page, 0,
                         state->rank, &state->local_mismatches);
    if (!local_ok) {
      break;
    }
  }
  query_memory(&state->free_owned, &state->total_hbm, "owned");

  if (!setup_control(state, socket_path)) {
    return ResultCode::kFailControl;
  }
  int32_t peer_status = 1;
  if (!exchange_status(state->control, state->rank, local_ok ? 0 : 1,
                       &peer_status, "LOCAL_VMM") ||
      !local_ok || peer_status != 0) {
    return local_ok ? ResultCode::kFailControl : ResultCode::kFailApi;
  }

  int32_t own_tgid = 0;
  if (!check_acl(aclrtDeviceGetBareTgid(&own_tgid),
                 "aclrtDeviceGetBareTgid")) {
    return ResultCode::kFailApi;
  }
  const PeerInfo own_info{kMagic, kVersion, state->rank, own_tgid,
                          state->own_device, 0, state->granularity};
  PeerInfo peer_info{};
  if (!exchange_value(state->control, state->rank, own_info, &peer_info) ||
      peer_info.magic != kMagic || peer_info.version != kVersion ||
      peer_info.rank != 1 - state->rank ||
      peer_info.granularity != state->granularity) {
    return ResultCode::kFailControl;
  }

  HandleBundle own_bundle{};
  own_bundle.magic = kMagic;
  own_bundle.version = kVersion;
  own_bundle.rank = state->rank;
  own_bundle.count = kOwnedPages;
  own_bundle.granularity = state->granularity;
  bool export_ok = true;
  for (int index = 0; index < kOwnedPages; ++index) {
    own_bundle.pages[index].logical_page = state->owned[index].logical_page;
    error = aclrtMemExportToShareableHandleV2(
        state->owned[index].handle, ACL_RT_VMM_EXPORT_FLAG_DEFAULT,
        ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
        &own_bundle.pages[index].handle);
    if (error == kCapabilityUnsupported) {
      return ResultCode::kBlockedCapability;
    }
    export_ok &= check_acl(error, "aclrtMemExportToShareableHandleV2");
    error = aclrtMemSetPidToShareableHandleV2(
        &own_bundle.pages[index].handle, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT,
        &peer_info.tgid, 1);
    if (error == kCapabilityUnsupported) {
      return ResultCode::kBlockedCapability;
    }
    export_ok &=
        check_acl(error, "aclrtMemSetPidToShareableHandleV2");
  }
  if (!exchange_status(state->control, state->rank, export_ok ? 0 : 1,
                       &peer_status, "EXPORTED") ||
      !export_ok || peer_status != 0) {
    return ResultCode::kFailApi;
  }

  HandleBundle peer_bundle{};
  if (!exchange_value(state->control, state->rank, own_bundle, &peer_bundle) ||
      peer_bundle.magic != kMagic || peer_bundle.version != kVersion ||
      peer_bundle.rank != 1 - state->rank ||
      peer_bundle.count != kOwnedPages ||
      peer_bundle.granularity != state->granularity) {
    return ResultCode::kFailControl;
  }

  bool import_ok = true;
  for (int index = 0; index < kOwnedPages; ++index) {
    PageState& page = state->imported[index];
    page.logical_page = peer_bundle.pages[index].logical_page;
    page.va = page_va(state->base, state->granularity, page.logical_page);
    error = aclrtMemImportFromShareableHandleV2(
        &peer_bundle.pages[index].handle, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, 0,
        &page.handle);
    if (error == kCapabilityUnsupported) {
      return ResultCode::kBlockedCapability;
    }
    import_ok &=
        check_acl(error, "aclrtMemImportFromShareableHandleV2");
    if (!import_ok) {
      break;
    }
    import_ok &=
        check_acl(aclrtMapMem(page.va, state->granularity, 0, page.handle, 0),
                  "aclrtMapMem imported");
    page.mapped = import_ok;
    if (!import_ok) {
      break;
    }
  }
  query_memory(&state->free_peak, &state->total_hbm, "peak");
  if (!exchange_status(state->control, state->rank, import_ok ? 0 : 1,
                       &peer_status, "MAPPED") ||
      !import_ok || peer_status != 0) {
    return ResultCode::kFailApi;
  }

  bool read_ok = true;
  for (int logical_page = 0; logical_page < kLogicalPages; ++logical_page) {
    const int owner_rank = logical_page % kWorldSize;
    size_t* mismatches =
        owner_rank == state->rank ? &state->local_mismatches
                                  : &state->peer_read_mismatches;
    read_ok &= validate_pattern(
        page_va(state->base, state->granularity, logical_page),
        state->granularity, logical_page, 0, owner_rank, mismatches);
  }
  if (!exchange_status(state->control, state->rank, read_ok ? 0 : 1,
                       &peer_status, "READ_OK") ||
      !read_ok || peer_status != 0) {
    return ResultCode::kFailData;
  }

  bool remote_write_ok = true;
  for (const auto& page : state->imported) {
    remote_write_ok &=
        write_pattern(page.va, state->granularity, page.logical_page, 1,
                      state->rank);
  }
  if (!exchange_status(state->control, state->rank,
                       remote_write_ok ? 0 : 1, &peer_status,
                       "REMOTE_WRITE_DONE") ||
      !remote_write_ok || peer_status != 0) {
    return ResultCode::kFailApi;
  }

  bool owner_observed_ok = true;
  for (const auto& page : state->owned) {
    owner_observed_ok &=
        validate_pattern(page.va, state->granularity, page.logical_page, 1,
                         1 - state->rank,
                         &state->remote_write_mismatches);
  }
  if (!exchange_status(state->control, state->rank,
                       owner_observed_ok ? 0 : 1, &peer_status,
                       "OWNER_OBSERVED") ||
      !owner_observed_ok || peer_status != 0) {
    return ResultCode::kFailData;
  }
  return ResultCode::kPass;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(
        stderr,
        "usage: %s <rank:0|1> <unix-socket> <own-device> <peer-device> "
        "<per-rank-budget-bytes>\n",
        argv[0]);
    return static_cast<int>(ResultCode::kFailApi);
  }
  ProbeState state{};
  state.rank = std::atoi(argv[1]);
  state.own_device = std::atoi(argv[3]);
  state.peer_device = std::atoi(argv[4]);
  const size_t budget =
      static_cast<size_t>(std::strtoull(argv[5], nullptr, 10));
  if (state.rank < 0 || state.rank >= kWorldSize ||
      state.own_device == state.peer_device || budget == 0) {
    std::fprintf(stderr, "FAIL invalid arguments\n");
    return static_cast<int>(ResultCode::kFailApi);
  }

  ResultCode result = run_probe(&state, argv[2], budget);
  result = cleanup(&state, result, argv[2]);
  print_json(state, result);
  return static_cast<int>(result);
}
