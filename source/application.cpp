#include "application.hpp"
#include "graphics.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // Глубина в клип-пространстве Vulkan — [0, 1]
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

namespace application {

namespace {

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

struct alignas(16) ObjectUniforms {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	glm::vec4 color;
};

constexpr uint32_t cylinder_segments = 50;
constexpr float cylinder_radius = 1.0f;
constexpr float cylinder_height = 2.0f;

graphics::Mesh cylinder_mesh;

VkDescriptorSetLayout descriptor_set_layout;
VkDescriptorPool descriptor_pool;
VkDescriptorSet descriptor_set;
graphics::Buffer uniform_buffer;

VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

glm::vec3 proceduralColor(const glm::vec3& p) {
	return glm::vec3(
		p.x / cylinder_radius * 0.5f + 0.5f,
		p.y / (cylinder_height * 0.5f) * 0.5f + 0.5f,
		p.z / cylinder_radius * 0.5f + 0.5f
	);
}

void buildCylinder(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                   uint32_t segments, float radius, float height) {
	const float half = height * 0.5f;

	// [0, segments) — верхнее кольцо, [segments, 2 * segments) — нижнее
	for (uint32_t i = 0; i < segments; ++i) {
		const float angle = glm::two_pi<float>() * float(i) / float(segments);
		const glm::vec3 top(radius * std::cos(angle), half, radius * std::sin(angle));
		vertices.push_back({ top, proceduralColor(top) });
	}
	for (uint32_t i = 0; i < segments; ++i) {
		const glm::vec3 bottom = vertices[i].position - glm::vec3(0.0f, height, 0.0f);
		vertices.push_back({ bottom, proceduralColor(bottom) });
	}

	const uint32_t top_center = uint32_t(vertices.size());
	vertices.push_back({ glm::vec3(0.0f, half, 0.0f), proceduralColor(glm::vec3(0.0f, half, 0.0f)) });
	const uint32_t bottom_center = uint32_t(vertices.size());
	vertices.push_back({ glm::vec3(0.0f, -half, 0.0f), proceduralColor(glm::vec3(0.0f, -half, 0.0f)) });

	for (uint32_t i = 0; i < segments; ++i) {
		const uint32_t j = (i + 1) % segments;
		const uint32_t top_i = i, top_j = j;
		const uint32_t bottom_i = segments + i, bottom_j = segments + j;

		// Боковая стенка: два треугольника на сегмент
		indices.insert(indices.end(), { bottom_i, top_i, bottom_j });
		indices.insert(indices.end(), { bottom_j, top_i, top_j });

		// Крышки: треугольные веера вокруг центров
		indices.insert(indices.end(), { top_center, top_j, top_i });
		indices.insert(indices.end(), { bottom_center, bottom_i, bottom_j });
	}
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout, nullptr,
	                                &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
	};

	const VkDescriptorPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	if (!graphics::createBuffer(uniform_buffer, sizeof(ObjectUniforms),
	                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
		return false;
	}

	const VkDescriptorSetAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &allocate, &descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}

	// Связываем запись 0 набора с uniform-буфером
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = uniform_buffer.buffer,
		.offset = 0,
		.range = sizeof(ObjectUniforms),
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	const VkPipelineLayoutCreateInfo layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &layout, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	const graphics::GraphicsPipelineDesc desc = {
		.vertex_shader_path = "shaders/cylinder.vert.spv",
		.fragment_shader_path = "shaders/cylinder.frag.spv",
		.vertex_bindings = {
			{
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			},
		},
		.vertex_attributes = {
			{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position),
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color),
			},
		},
		.layout = pipeline_layout,
		.render_pass = context.render_pass,
	};

	pipeline = graphics::createGraphicsPipeline(desc);
	return pipeline != VK_NULL_HANDLE;
}

} // namespace

// ---------------------------------------------------------------------------

bool initialize() {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	buildCylinder(vertices, indices, cylinder_segments, cylinder_radius, cylinder_height);

	if (!graphics::createMesh(cylinder_mesh, vertices, indices)) {
		return false;
	}

	return createDescriptors() && createPipeline();
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	graphics::destroyBuffer(uniform_buffer);
	// Набор дескрипторов освобождается вместе с пулом
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);

	graphics::destroyMesh(cylinder_mesh);
}

void update([[maybe_unused]] double time) {
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	auto& context = graphics::internal::context;
	const VkExtent2D extent = context.swapchain_extent;

	// Пока точка наблюдения фиксированная, матрицы уедут в интерфейс позже
	const float aspect = float(extent.width) / float(extent.height);
	glm::mat4 projection = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
	// GLM рассчитан на OpenGL (Y в клип-пространстве вверх), в Vulkan — вниз
	projection[1][1] *= -1.0f;

	const ObjectUniforms uniforms = {
		.model = glm::mat4(1.0f),
		.view = glm::lookAt(glm::vec3(4.0f, 3.0f, 7.5f), glm::vec3(0.0f),
		                    glm::vec3(0.0f, 1.0f, 0.0f)),
		.projection = projection,
		.color = glm::vec4(1.0f),
	};
	graphics::uploadBuffer(uniform_buffer, &uniforms, sizeof(uniforms));

	VkCommandBuffer cmd = fd.command_buffer;

	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin);

	const VkClearValue clear_values[] = {
		{ .color = { { 0.08f, 0.09f, 0.12f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .offset = { 0, 0 }, .extent = extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cmd, &render_pass, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(extent.width),
		.height = float(extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	const VkRect2D scissor = { .offset = { 0, 0 }, .extent = extent };
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &cylinder_mesh.vertex_buffer.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, cylinder_mesh.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &descriptor_set, 0, nullptr);
	vkCmdDrawIndexed(cmd, cylinder_mesh.index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);}

} // namespace application
