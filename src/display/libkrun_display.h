#ifndef _LIBKRUN_DISPLAY_H
#define _LIBKRUN_DISPLAY_H

#include <inttypes.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// The display backend encountered an internal error
#define KRUN_DISPLAY_ERR_INTERNAL -1
#define KRUN_DISPLAY_ERR_METHOD_UNSUPPORTED -2
#define KRUN_DISPLAY_ERR_INVALID_SCANOUT_ID -3
#define KRUN_DISPLAY_ERR_INVALID_PARAM -4
#define KRUN_DISPLAY_ERR_OUT_OF_BUFFERS -5

// Same as VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM
#define KRUN_DISPLAY_FORMAT_B8G8R8A8_UNORM 1
// Same as VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM
#define KRUN_DISPLAY_FORMAT_B8G8R8X8_UNORM 2
// Same as VIRTIO_GPU_FORMAT_A8R8G8B8_UNORM
#define KRUN_DISPLAY_FORMAT_A8R8G8B8_UNORM 3
// Same as VIRTIO_GPU_FORMAT_X8R8G8B8_UNORM
#define KRUN_DISPLAY_FORMAT_X8R8G8B8_UNORM 4
// Same as VIRTIO_GPU_FORMAT_R8G8B8A8_UNORM
#define KRUN_DISPLAY_FORMAT_R8G8B8A8_UNORM 67
// Same as VIRTIO_GPU_PIXEL_FORMAT_X8B8G8R8_UNORM
#define KRUN_DISPLAY_FORMAT_X8B8G8R8_UNORM 68
// Same as VIRTIO_GPU_PIXEL_FORMAT_A8B8G8R8_UNORM
#define KRUN_DISPLAY_FORMAT_A8B8G8R8_UNORM 121
// Same as VIRTIO_GPU_PIXEL_FORMAT_R8G8B8X8_UNORM
#define KRUN_DISPLAY_FORMAT_R8G8B8X8_UNORM 134

/**
 * Indicates support for basic framebuffer operations.
 * If supported, the implementation must provide `disable_scanout`, `configure_scanout`, `alloc_frame`,
 * and `present_frame`.
 */
#define KRUN_DISPLAY_FEATURE_BASIC_FRAMEBUFFER 1

/**
 * Called to create a display instance.
 *
 * Arguments:
 *  "instance"    - (Output) pointer to userdata which can be used to represents this/self argument.
 *                  Implementation may set it to any value (even NULL)
 *  "userdata"    - userdata specified in the `krun_display_backend` instance
 *  "reserved"    - reserved/unused for now
 *
 * Returns:
 *  Zero on success or a negative error code (KRUN_DISPLAY_ERR_*) otherwise.
 */
typedef int32_t (*krun_display_create_fn)(void **instance, const void *userdata, const void *reserved);

/**
 * Called to destroy the display instance.
 *
 * Arguments:
 *  "instance"    - userdata set by `krun_display_create`, represents this/self argument
 *
 * Returns:
 *  Zero on success or a negative error code (KRUN_DISPLAY_ERR_*) otherwise.
 */
typedef int32_t (*krun_display_destroy_fn)(void *instance);

/**
 * Configures or reconfigures a display scanout.
 *
 * Arguments:
 *  "instance"       - userdata set by `krun_display_create`, represents this/self argument
 *  "scanout_id"     - The identifier of the scanout to configure.
 *  "display_width"  - The original width of the display in pixels.
 *  "display_height" - The original height of the display in pixels.
 *  "width"          - The width of the configured scanout in pixels.
 *  "height"         - The height of the configured scanout in pixels.
 *  "format"         - The pixel format for the scanout (see KRUN_DISPLAY_FORMAT_* constants).
 *
 * Returns:
 *  Zero on success or a negative error code (KRUN_DISPLAY_ERR_*) otherwise.
 */
typedef int32_t (*krun_display_configure_scanout_fn)(void *instance,
    uint32_t scanout_id,
    uint32_t display_width,
    uint32_t display_height,
    uint32_t width,
    uint32_t height,
    uint32_t format);

/**
 * Disables a display scanout.
 *
 * Arguments:
 *  "instance"    - userdata set by `krun_display_create`, represents this/self argument
 *  "scanout_id"  - The identifier of the scanout to disable.
 *
 * Returns:
 *  Zero on success or a negative error code (KRUN_DISPLAY_ERR_*) otherwise.
 */
typedef int32_t (*krun_display_disable_scanout_fn)(void *instance, uint32_t scanout_id);

/**
 * Allocates a new frame for a specified scanout.
 * This function provides a direct pointer to the frame's buffer.
 * The caller is responsible for writing pixel data into this buffer.
 *
 * Arguments:
 *  "instance"    - userdata set by `krun_display_create`, represents this/self argument
 *  "scanout_id"  - The identifier of the scanout for which to allocate the frame.
 *  "buffer"      - (Output) A pointer to a pointer that will be set to the address
 *                  of the allocated frame's memory. The memory pointed to
 *                  by *buffer must be writable by the caller.
 * "buffer_size"  -  (Output) The size of the allocated buffer. This is mostly a sanity check, because the size
 *                   is already determined by krun_display_configure_scanout.
 *
 * Returns:
 *  The "frame_id" of the allocated frame or a negative error code (KRUN_DISPLAY_ERR_*) otherwise.
 */
typedef int32_t (*krun_display_alloc_frame_fn)(void *instance, uint32_t scanout_id, uint8_t **buffer, size_t *buffer_size);

struct krun_rect {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
};

/**
 * Presents a previously allocated frame to the display.
 * After this call, the `frame_id` is considered consumed or "deallocated"
 * from the user's perspective. The user must call `krun_display_alloc_frame`
 * again to obtain a new valid frame for the next rendering cycle.
 * The content of the buffer associated with the `frame_id` should not be
 * modified after this call.
 *
 * Arguments:
 *  "instance"        - userdata set by `krun_display_create`, represents this/self argument
 *  "scanout_id"      - The identifier of the scanout on which to present the frame.
 *  "frame_id"        - The identifier of the frame to present, previously obtained from `krun_display_alloc_frame`.
* "damage_area"       - (Optional) Optimization hint describing the area that has changed since the last call to
 *                      present_frame. If NULL, the entire frame is assumed to be damaged.
 *
 * Returns:
 * Zero on success or a negative error or a negative error code (KRUN_DISPLAY_ERR_*) otherwise.
 */
typedef int32_t (*krun_display_present_frame_fn)(void *instance, uint32_t scanout_id, uint32_t frame_id, const struct krun_rect* damage_area);

/**
 * (limina extension, optional) Sets the hardware cursor image and hotspot.
 *
 * Called when the guest issues VIRTIO_GPU_CMD_UPDATE_CURSOR. The backend should render this
 * image as an overlay (NOT into the scanout) at the position last given by move_cursor, with
 * the hotspot subtracted. A width/height of 0 (or a NULL buffer) means hide the cursor.
 *
 * Arguments:
 *  "instance"    - userdata set by `krun_display_create`, represents this/self argument
 *  "scanout_id"  - which scanout the cursor belongs to. Load-bearing on a multi-head guest: a
 *                  compositor enables the cursor plane on only the CRTC the pointer is on and
 *                  disables the others, so this is the whole signal for which display shows it.
 *  "width"       - cursor image width in pixels (0 to hide)
 *  "height"      - cursor image height in pixels (0 to hide)
 *  "hot_x"       - cursor hotspot x within the image
 *  "hot_y"       - cursor hotspot y within the image
 *  "format"      - pixel format of `buffer` (see KRUN_DISPLAY_FORMAT_* constants)
 *  "buffer"      - cursor pixels, `width * height * 4` bytes in `format` (NULL to hide)
 *  "buffer_size" - length of `buffer` in bytes
 *
 * Returns:
 *  Zero on success or a negative error code (KRUN_DISPLAY_ERR_*) otherwise.
 */
typedef int32_t (*krun_display_set_cursor_fn)(void *instance,
    uint32_t scanout_id,
    uint32_t width,
    uint32_t height,
    uint32_t hot_x,
    uint32_t hot_y,
    uint32_t format,
    const uint8_t *buffer,
    size_t buffer_size);

/**
 * (limina extension, optional) Moves the hardware cursor.
 *
 * Called when the guest issues VIRTIO_GPU_CMD_MOVE_CURSOR (and on UPDATE_CURSOR). The
 * position is in scanout pixels; the backend applies the hotspot from the last set_cursor.
 *
 * Arguments:
 *  "instance"    - userdata set by `krun_display_create`, represents this/self argument
 *  "scanout_id"  - which scanout `x`/`y` are relative to
 *  "x"           - cursor x position in scanout pixels
 *  "y"           - cursor y position in scanout pixels
 *
 * Returns:
 *  Zero on success or a negative error code (KRUN_DISPLAY_ERR_*) otherwise.
 */
typedef int32_t (*krun_display_move_cursor_fn)(void *instance, uint32_t scanout_id, uint32_t x, uint32_t y);

/**
 * (limina extension, optional) Presents an externally-rendered surface to the display zero-copy.
 *
 * Called for VIRTIO_GPU_CMD_SET_SCANOUT_BLOB scanouts whose resource is backed by a global
 * IOSurface (venus rendered straight into it). The backend presents that IOSurface directly
 * by its global id (e.g. forwarding it to the supervisor, which IOSurfaceLookup()s it) — no
 * alloc_frame/copy. Unlike present_frame there is no prior alloc_frame; the surface is owned
 * by the renderer, not the backend. A backend without zero-copy support returns
 * KRUN_DISPLAY_ERR_METHOD_UNSUPPORTED and the device falls back to the readback path.
 *
 * Arguments:
 *  "instance"     - userdata set by `krun_display_create`, represents this/self argument
 *  "scanout_id"   - The identifier of the scanout on which to present.
 *  "iosurface_id" - The global IOSurface id (IOSurfaceGetID) to present.
 *  "damage_area"  - (Optional) changed-area hint; NULL means the whole surface is damaged.
 *
 * Returns:
 *  Zero on success or a negative error code (KRUN_DISPLAY_ERR_*) otherwise.
 */
typedef int32_t (*krun_display_present_surface_fn)(void *instance, uint32_t scanout_id, uint32_t iosurface_id, const struct krun_rect* damage_area);

/**
 * (limina extension, optional) The guest dropped its last reference to a scanout resource.
 *
 * Called from VIRTIO_GPU_CMD_RESOURCE_UNREF for a resource backed by an IOSurface the backend has
 * been given (one previously passed to present_surface). After this call that IOSurface will never
 * be presented again, so a backend holding it — or having handed it to another process — may drop it.
 *
 * This exists because that reference is otherwise unbounded in practice. A compositor is free to
 * allocate a fresh scanout buffer per frame, and a backend that caches by IOSurface id then
 * accumulates one whole framebuffer per frame with nothing to tell it otherwise. On macOS the
 * storage bills to the task that CREATED the surface, so the holder feels no pressure of its own.
 * The guest's unref is the only accurate signal that a surface is done with.
 *
 * Arguments:
 *  "instance"     - userdata set by `krun_display_create`, represents this/self argument
 *  "iosurface_id" - The IOSurface id (IOSurfaceGetID) the released resource was backed by.
 *
 * Returns:
 *  Zero on success or a negative error code (KRUN_DISPLAY_ERR_*) otherwise. The device ignores the
 *  result: the resource is freed either way.
 */
typedef int32_t (*krun_display_release_surface_fn)(void *instance, uint32_t iosurface_id);

/**
 * (limina extension, optional) A guest OS driver has taken over the GPU device.
 *
 * Called once per VM run, the first time the guest asks for a display's EDID
 * (VIRTIO_GPU_CMD_GET_EDID). Boot firmware drives the device before this — it programs scanout 0
 * and never reads an EDID — so this call is the boundary between the firmware's use of the device
 * and the operating system's.
 *
 * A backend that arranges displays needs that boundary, because the two consumers have different
 * constraints: firmware drivers commonly hardcode scanout 0 and cannot be given a different one,
 * while an OS driver enumerates every scanout and follows connector hotplug. Without the signal a
 * backend must either apply the firmware's constraint forever or guess when it has been lifted.
 *
 * Arguments:
 *  "instance"    - userdata set by `krun_display_create`, represents this/self argument
 *
 * Returns:
 *  Zero on success or a negative error code (KRUN_DISPLAY_ERR_*) otherwise. The device ignores
 *  the result.
 */
typedef int32_t (*krun_display_guest_driver_ready_fn)(void *instance);

/**
 * Defines the set of callbacks for a display implementation.
 * This structure holds function pointers that a display backend implements to integrate with the libkrun.
 *
 * This is modeled as an object, an object instance is created using the `create` function and destroyed using `destroy`.
 * It is possible for the `create` function to be null in this case, the pointer to the object instance will be null
 * in the methods.
 *
 * The gpu device instantiates the display backend using the krun_display_create in a specific thread. All further calls
 * to the display backend will be called from the same thread. Note that the display methods should not block for a long
 * time otherwise this will negatively impact performance of the emulated GPU device.
 *
 * See krun_display_* function pointer typedef definitions for descriptions of individual methods.
 * In the future more methods may be added, depending on which KRUN_DISPLAY_FEATURE_* flags are passed to
 * krun_set_display_backend. The user of the library *MUST* zero initialize this struct to make all (future) unset
 * fields NULL.
 */
struct krun_display_basic_framebuffer_vtable {
    krun_display_destroy_fn             destroy; // (optional)
    krun_display_disable_scanout_fn     disable_scanout; // Required by KRUN_DISPLAY_FEATURE_BASIC_FRAMEBUFFER
    krun_display_configure_scanout_fn   configure_scanout; // Required by KRUN_DISPLAY_FEATURE_BASIC_FRAMEBUFFER
    krun_display_alloc_frame_fn         alloc_frame; // Required by KRUN_DISPLAY_FEATURE_BASIC_FRAMEBUFFER
    krun_display_present_frame_fn       present_frame; // Required by KRUN_DISPLAY_FEATURE_BASIC_FRAMEBUFFER
    krun_display_set_cursor_fn          set_cursor; // (optional) limina: hardware cursor image+hotspot
    krun_display_move_cursor_fn         move_cursor; // (optional) limina: hardware cursor position
    krun_display_present_surface_fn     present_surface; // (optional) limina: zero-copy IOSurface scanout present
    krun_display_release_surface_fn     release_surface; // (optional) limina: guest unref'd a scanout resource
    krun_display_guest_driver_ready_fn  guest_driver_ready; // (optional) limina: OS driver took over
};

union krun_display_vtable {
    struct krun_display_basic_framebuffer_vtable basic_framebuffer;
};

struct krun_display_backend {
    uint64_t features;
    void *create_userdata; // (optional)
    krun_display_create_fn create; // (optional)
    union krun_display_vtable vtable;
};

#ifdef __cplusplus
}
#endif

#endif // _LIBKRUN_DISPLAY_H
