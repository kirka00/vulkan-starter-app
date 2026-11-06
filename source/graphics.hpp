#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

#include <vk_mem_alloc.h>

namespace graphics {

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr; // nullptr у буферов в VRAM
	VkDeviceSize size = 0;
};

// Буфер в памяти, видимой CPU: uniform, storage и staging
bool createBuffer(Buffer& out, VkDeviceSize size, VkBufferUsageFlags usage,
                  const void* initial_data = nullptr);
// Буфер в VRAM: CPU в него не пишет, данные приходят только командой копирования
bool createDeviceBuffer(Buffer& out, VkDeviceSize size, VkBufferUsageFlags usage);
void uploadBuffer(Buffer& buffer, const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
void destroyBuffer(Buffer& buffer);

struct Texture {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t mip_levels = 0;
};

// Пакет копирований в VRAM: данные складываются в один staging-буфер, команды
// копирования пишутся в одноразовый командный буфер и отправляются одним submit
struct Uploader {
	Buffer staging;
	VkDeviceSize offset = 0;
	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkCommandBuffer command_buffer = VK_NULL_HANDLE;
};

bool beginUpload(Uploader& up, VkDeviceSize capacity);
bool uploadToBuffer(Uploader& up, const Buffer& dst, const void* data, VkDeviceSize size);
bool uploadToTexture(Uploader& up, Texture& out, const void* pixels,
                     uint32_t width, uint32_t height, VkFormat format);
bool endUpload(Uploader& up);

// Сколько места займут данные в staging-буфере вместе с выравниванием
VkDeviceSize uploadFootprint(VkDeviceSize size);

void destroyTexture(Texture& texture);

struct Mesh {
	Buffer vertex_buffer;
	Buffer index_buffer;
	uint32_t vertex_count = 0;
	uint32_t index_count = 0;
};

template <typename Vertex>
bool createMesh(Uploader& up, Mesh& out, const std::vector<Vertex>& vertices,
                const std::vector<uint32_t>& indices) {
	const VkDeviceSize vertices_size = sizeof(Vertex) * vertices.size();
	const VkDeviceSize indices_size = sizeof(uint32_t) * indices.size();

	if (!createDeviceBuffer(out.vertex_buffer, vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
	    !createDeviceBuffer(out.index_buffer, indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) {
		return false;
	}

	if (!uploadToBuffer(up, out.vertex_buffer, vertices.data(), vertices_size) ||
	    !uploadToBuffer(up, out.index_buffer, indices.data(), indices_size)) {
		return false;
	}

	out.vertex_count = uint32_t(vertices.size());
	out.index_count = uint32_t(indices.size());
	return true;
}

void destroyMesh(Mesh& mesh);

VkShaderModule loadShaderModule(const std::string& path);

struct GraphicsPipelineDesc {
	std::string vertex_shader_path;
	std::string fragment_shader_path;

	std::vector<VkVertexInputBindingDescription> vertex_bindings;
	std::vector<VkVertexInputAttributeDescription> vertex_attributes;

	VkPipelineLayout layout = VK_NULL_HANDLE;
	VkRenderPass render_pass = VK_NULL_HANDLE;

	VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
	VkCullModeFlags cull_mode = VK_CULL_MODE_BACK_BIT;
	VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;

	bool depth_test = true;
	bool depth_write = true;
};

VkPipeline createGraphicsPipeline(const GraphicsPipelineDesc& desc);

} // namespace graphics
