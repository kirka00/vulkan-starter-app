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
	void* mapped = nullptr;
	VkDeviceSize size = 0;
};

bool createBuffer(Buffer& out, VkDeviceSize size, VkBufferUsageFlags usage,
                  const void* initial_data = nullptr);
void uploadBuffer(Buffer& buffer, const void* data, VkDeviceSize size);
void destroyBuffer(Buffer& buffer);

struct Mesh {
	Buffer vertex_buffer;
	Buffer index_buffer;
	uint32_t vertex_count = 0;
	uint32_t index_count = 0;
};

template <typename Vertex>
bool createMesh(Mesh& out, const std::vector<Vertex>& vertices,
                const std::vector<uint32_t>& indices) {
	if (!createBuffer(out.vertex_buffer, sizeof(Vertex) * vertices.size(),
	                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data())) {
		return false;
	}

	if (!createBuffer(out.index_buffer, sizeof(uint32_t) * indices.size(),
	                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data())) {
		destroyBuffer(out.vertex_buffer);
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
