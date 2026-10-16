#include "application.hpp"
#include "graphics.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include <imgui.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // Глубина в клип-пространстве Vulkan — [0, 1]
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

namespace application {

namespace {

// ---------------------------------------------------------------------------
// Структуры, общие с шейдерами

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

// ---------------------------------------------------------------------------
// Описание сцены

struct Transform {
	glm::vec3 position = glm::vec3(0.0f);
	glm::vec3 rotation = glm::vec3(0.0f); // Углы Эйлера, градусы
	glm::vec3 scale = glm::vec3(1.0f);
};

struct Animation {
	bool enabled = true;
	bool playing = true;
	float time = 0.0f;   // Собственное время анимации, секунды
	float speed = 1.0f;  // Множитель скорости
	float radius = 2.0f;
	float height = 1.0f;
	float loops = 3.0f;
	float spin = 90.0f;  // Скорость собственного вращения, градусов в секунду
};

struct Camera {
	bool perspective = true;
	float fov = 60.0f;          // Градусы, только для перспективы
	float ortho_height = 10.0f; // Видимая высота в мировых единицах, только для ортографии

	// Точка наблюдения фиксированная: подвижная камера не входит в эту работу
	static constexpr float near_plane = 0.1f;
	static constexpr float far_plane = 100.0f;
	static constexpr float distance = 9.0f;
	static constexpr float yaw = 30.0f;   // Градусы
	static constexpr float pitch = 20.0f; // Градусы
};

constexpr uint32_t cylinder_segments = 50;
constexpr float cylinder_radius = 1.0f;
constexpr float cylinder_height = 2.0f;

// ---------------------------------------------------------------------------
// Состояние

graphics::Mesh cylinder_mesh;

VkDescriptorSetLayout descriptor_set_layout;
VkDescriptorPool descriptor_pool;
VkDescriptorSet descriptor_set;
graphics::Buffer uniform_buffer;

VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

Transform transform;
Animation animation;
glm::vec4 color = glm::vec4(1.0f, 0.85f, 0.6f, 1.0f);
Camera camera;

double last_time = 0.0;

// ---------------------------------------------------------------------------
// Геометрия

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

// ---------------------------------------------------------------------------
// Математика

glm::mat4 modelMatrix(const Transform& t, const Animation& a) {
	glm::vec3 position = t.position;
	glm::mat4 rotation = glm::mat4(1.0f);

	if (a.enabled) {
		const float angle = a.time;
		position += glm::vec3(a.radius * std::cos(angle),
		                      a.height * std::sin(a.loops * angle),
		                      a.radius * std::sin(angle));

		const float spin = glm::radians(a.spin) * a.time;
		rotation = glm::rotate(glm::mat4(1.0f), spin, glm::vec3(0.0f, 1.0f, 0.0f)) *
		           glm::rotate(glm::mat4(1.0f), spin * 0.5f, glm::vec3(1.0f, 0.0f, 0.0f));
	}

	rotation = rotation *
	           glm::rotate(glm::mat4(1.0f), glm::radians(t.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f)) *
	           glm::rotate(glm::mat4(1.0f), glm::radians(t.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f)) *
	           glm::rotate(glm::mat4(1.0f), glm::radians(t.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));

	return glm::translate(glm::mat4(1.0f), position) * rotation *
	       glm::scale(glm::mat4(1.0f), t.scale);
}

glm::mat4 viewMatrix(const Camera& c) {
	const float yaw = glm::radians(c.yaw);
	const float pitch = glm::radians(c.pitch);
	const glm::vec3 eye = c.distance * glm::vec3(
		std::cos(pitch) * std::sin(yaw),
		std::sin(pitch),
		std::cos(pitch) * std::cos(yaw)
	);
	return glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 projectionMatrix(const Camera& c, float aspect) {
	glm::mat4 projection;
	if (c.perspective) {
		projection = glm::perspective(glm::radians(c.fov), aspect, c.near_plane, c.far_plane);
	} else {
		const float half_h = c.ortho_height * 0.5f;
		const float half_w = half_h * aspect;
		projection = glm::ortho(-half_w, half_w, -half_h, half_h, c.near_plane, c.far_plane);
	}
	// GLM рассчитан на OpenGL (Y в клип-пространстве вверх), в Vulkan — вниз
	projection[1][1] *= -1.0f;
	return projection;
}

// ---------------------------------------------------------------------------
// Объекты Vulkan

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

// ---------------------------------------------------------------------------
// Интерфейс

void drawUI() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 1 - Cylinder");

	ImGui::Text("%.1f FPS  |  %u vertices, %u triangles",
	            double(ImGui::GetIO().Framerate), cylinder_mesh.vertex_count,
	            cylinder_mesh.index_count / 3);

	if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		int mode = camera.perspective ? 0 : 1;
		ImGui::RadioButton("Perspective", &mode, 0);
		ImGui::SameLine();
		ImGui::RadioButton("Orthographic", &mode, 1);
		camera.perspective = (mode == 0);

		if (camera.perspective) {
			ImGui::SliderFloat("FOV (deg)", &camera.fov, 10.0f, 150.0f);
		} else {
			ImGui::SliderFloat("View height", &camera.ortho_height, 1.0f, 30.0f);
		}
	}

	if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position", &transform.position.x, 0.02f);
		ImGui::DragFloat3("Rotation (deg)", &transform.rotation.x, 1.0f, -360.0f, 360.0f);
		ImGui::DragFloat3("Scale", &transform.scale.x, 0.02f, 0.05f, 10.0f);
	}

	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Enable trajectory", &animation.enabled);
		ImGui::BeginDisabled(!animation.enabled);
		if (ImGui::Button(animation.playing ? "Pause" : "Play")) {
			animation.playing = !animation.playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("Restart")) {
			animation.time = 0.0f;
		}
		ImGui::SameLine();
		ImGui::Text("t = %.2f s", animation.time);
		ImGui::SliderFloat("Speed", &animation.speed, 0.0f, 5.0f);
		ImGui::SliderFloat("Radius", &animation.radius, 0.0f, 6.0f);
		ImGui::SliderFloat("Height", &animation.height, 0.0f, 4.0f);
		ImGui::SliderFloat("Loops per revolution", &animation.loops, 0.0f, 8.0f);
		ImGui::SliderFloat("Spin (deg/s)", &animation.spin, -360.0f, 360.0f);
		ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Color", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Tint", &color.x);
	}

	ImGui::End();
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

	if (!createDescriptors() || !createPipeline()) {
		return false;
	}

	transform.rotation = glm::vec3(20.0f, 0.0f, 15.0f);

	return true;
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

void update(double time) {
	const float dt = last_time > 0.0 ? float(time - last_time) : 0.0f;
	last_time = time;

	if (animation.enabled && animation.playing) {
		animation.time += dt * animation.speed;
	}

	drawUI();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	auto& context = graphics::internal::context;
	const VkExtent2D extent = context.swapchain_extent;

	// Предыдущий кадр уже дорисован (prepare дождался fence),
	// поэтому uniform-буфер можно спокойно перезаписать
	const float aspect = float(extent.width) / float(extent.height);
	const ObjectUniforms uniforms = {
		.model = modelMatrix(transform, animation),
		.view = viewMatrix(camera),
		.projection = projectionMatrix(camera, aspect),
		.color = color,
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
	vkEndCommandBuffer(cmd);
}

} // namespace application
