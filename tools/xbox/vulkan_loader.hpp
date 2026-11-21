#ifndef TOOLS_XBOX_VULKAN_LOADER_HPP
#define TOOLS_XBOX_VULKAN_LOADER_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define XBOX_VK_GLOBAL_FUNCTIONS(X) \
    X(vkCreateInstance)

#define XBOX_VK_INSTANCE_FUNCTIONS(X) \
    X(vkAllocateCommandBuffers) X(vkAllocateDescriptorSets) X(vkAllocateMemory) X(vkBeginCommandBuffer) X(vkBindBufferMemory) \
    X(vkBindImageMemory) X(vkCmdBeginRendering) X(vkCmdBindDescriptorSets) X(vkCmdBindPipeline) X(vkCmdClearColorImage) \
    X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdDispatch) X(vkCmdDraw) X(vkCmdEndRendering) X(vkCmdPipelineBarrier) \
    X(vkCmdPushConstants) X(vkCreateBuffer) X(vkCreateCommandPool) X(vkCreateComputePipelines) X(vkCreateDescriptorPool) \
    X(vkCreateDescriptorSetLayout) X(vkCreateDevice) X(vkCreateGraphicsPipelines) X(vkCreateImage) X(vkCreateImageView) \
    X(vkCreatePipelineLayout) X(vkCreateSampler) X(vkCreateShaderModule) X(vkEndCommandBuffer) X(vkEnumeratePhysicalDevices) \
    X(vkGetBufferMemoryRequirements) X(vkGetDeviceProcAddr) X(vkGetDeviceQueue) X(vkGetImageMemoryRequirements) \
    X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceFormatProperties) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkMapMemory) \
    X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkUpdateDescriptorSets)

#define XBOX_VK_DECLARE(name) inline PFN_##name name = nullptr;
inline PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
XBOX_VK_GLOBAL_FUNCTIONS(XBOX_VK_DECLARE)
XBOX_VK_INSTANCE_FUNCTIONS(XBOX_VK_DECLARE)
#undef XBOX_VK_DECLARE

inline void LoadVulkanLoader() {
    if (vkGetInstanceProcAddr != nullptr) return;
#ifdef _WIN32
    HMODULE library = LoadLibraryA("vulkan-1.dll");
    if (library == nullptr) throw std::runtime_error("cannot load vulkan-1.dll");
    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
#else
    void* library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) throw std::runtime_error(std::string("cannot load the Vulkan loader: ") + dlerror());
    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
    if (vkGetInstanceProcAddr == nullptr) throw std::runtime_error("the Vulkan loader has no vkGetInstanceProcAddr");
#define XBOX_VK_LOAD(name) name = reinterpret_cast<PFN_##name>(vkGetInstanceProcAddr(VK_NULL_HANDLE, #name)); \
    if (name == nullptr) throw std::runtime_error("the Vulkan loader has no " #name);
    XBOX_VK_GLOBAL_FUNCTIONS(XBOX_VK_LOAD)
#undef XBOX_VK_LOAD
}

inline void LoadVulkanInstance(VkInstance instance) {
#define XBOX_VK_LOAD(name) name = reinterpret_cast<PFN_##name>(vkGetInstanceProcAddr(instance, #name)); \
    if (name == nullptr) throw std::runtime_error("the Vulkan instance has no " #name);
    XBOX_VK_INSTANCE_FUNCTIONS(XBOX_VK_LOAD)
#undef XBOX_VK_LOAD
}

#endif
