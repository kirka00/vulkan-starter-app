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

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
};

struct alignas(16) SceneUniforms {
	glm::mat4 view_projection;
	glm::vec4 camera_position;
	glm::vec4 ambient;        // rgb = цвет, a = интенсивность
	glm::vec4 sun_direction;  // xyz = направление, куда летит свет
	glm::vec4 sun_color;      // rgb = цвет, a = интенсивность
	glm::uvec4 light_counts;  // x = точечные источники, y = прожекторные
};

struct alignas(16) ModelUniforms {
	glm::mat4 model;
	glm::mat4 normal_matrix;
	glm::vec4 albedo;
	glm::vec4 specular; // rgb = цвет блика, a = коэффициент блеска
};

struct alignas(16) PointLightGPU {
	glm::vec4 position;
	glm::vec4 color;
};

struct alignas(16) SpotLightGPU {
	glm::vec4 position;  // w = косинус внутреннего угла
	glm::vec4 direction; // w = косинус внешнего угла
	glm::vec4 color;
};

// ---------------------------------------------------------------------------

struct Camera {
	glm::vec3 position = glm::vec3(0.0f, 1.2f, 7.0f);
	float yaw = 0.0f;    // Градусы, поворот вокруг Y; при 0 взгляд вдоль -Z
	float pitch = -6.0f; // Градусы
	float fov = 60.0f;

	static constexpr float move_speed = 5.0f;
	static constexpr float near_plane = 0.1f;
	static constexpr float far_plane = 100.0f;
};

struct Material {
	glm::vec3 albedo = glm::vec3(0.85f, 0.50f, 0.25f);
	glm::vec3 specular = glm::vec3(1.0f);
	float shininess = 48.0f;
};

struct DirectionalLight {
	glm::vec3 direction = glm::vec3(-0.55f, -0.75f, -0.45f);
	glm::vec3 color = glm::vec3(1.0f, 0.96f, 0.88f);
	float intensity = 0.5f;
};

struct PointLight {
	glm::vec3 position = glm::vec3(2.5f, 0.6f, 2.0f);
	glm::vec3 color = glm::vec3(1.0f, 0.35f, 0.15f);
	float intensity = 14.0f;
};

struct SpotLight {
	glm::vec3 position = glm::vec3(0.0f, 4.5f, 1.0f);
	glm::vec3 direction = glm::vec3(0.0f, -1.0f, -0.15f);
	glm::vec3 color = glm::vec3(0.45f, 0.75f, 1.0f);
	float intensity = 55.0f;
	float inner_angle = 14.0f; // Градусы, половина угла яркой середины
	float outer_angle = 24.0f; // Градусы, половина угла, где свет гаснет
};

struct Model {
	graphics::Mesh mesh;
	graphics::Buffer uniform_buffer;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

constexpr uint32_t cylinder_segments = 50;
constexpr float cylinder_radius = 1.0f;
constexpr float cylinder_height = 2.0f;

constexpr float floor_half_size = 7.0f;
constexpr float floor_level = -1.5f;
constexpr float cylinder_level = floor_level + cylinder_height * 0.5f;

constexpr uint32_t light_capacity = 16;

// ---------------------------------------------------------------------------
// Состояние

Model cylinder;
Model floor_plane;

VkDescriptorSetLayout scene_set_layout;
VkDescriptorSetLayout model_set_layout;
VkDescriptorPool descriptor_pool;

VkDescriptorSet scene_descriptor_set;
graphics::Buffer scene_uniform_buffer;
graphics::Buffer point_light_buffer;
graphics::Buffer spot_light_buffer;

VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

Camera camera;
Material cylinder_material;
const Material floor_material = { glm::vec3(0.22f, 0.24f, 0.28f), glm::vec3(0.35f), 16.0f };
DirectionalLight sun;
glm::vec3 ambient_color = glm::vec3(1.0f);
float ambient_intensity = 0.05f;

std::vector<PointLight> point_lights = { PointLight{} };
std::vector<SpotLight> spot_lights = { SpotLight{} };

double last_time = 0.0;

// ---------------------------------------------------------------------------
// Геометрия

void buildCylinder(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                   uint32_t segments, float radius, float height) {
	const float half = height * 0.5f;

	// [0, N) — верх стенки, [N, 2N) — низ стенки
	for (uint32_t i = 0; i < segments; ++i) {
		const float angle = glm::two_pi<float>() * float(i) / float(segments);
		const glm::vec3 normal(std::cos(angle), 0.0f, std::sin(angle));
		vertices.push_back({ glm::vec3(radius * normal.x, half, radius * normal.z), normal });
	}
	for (uint32_t i = 0; i < segments; ++i) {
		const glm::vec3 normal = vertices[i].normal;
		vertices.push_back({ glm::vec3(radius * normal.x, -half, radius * normal.z), normal });
	}

	// [2N, 3N) — кольцо верхней крышки, затем её центр
	const uint32_t top_ring = uint32_t(vertices.size());
	for (uint32_t i = 0; i < segments; ++i) {
		vertices.push_back({ glm::vec3(vertices[i].position.x, half, vertices[i].position.z),
		                     glm::vec3(0.0f, 1.0f, 0.0f) });
	}
	const uint32_t top_center = uint32_t(vertices.size());
	vertices.push_back({ glm::vec3(0.0f, half, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f) });

	// Кольцо нижней крышки, затем её центр
	const uint32_t bottom_ring = uint32_t(vertices.size());
	for (uint32_t i = 0; i < segments; ++i) {
		vertices.push_back({ glm::vec3(vertices[i].position.x, -half, vertices[i].position.z),
		                     glm::vec3(0.0f, -1.0f, 0.0f) });
	}
	const uint32_t bottom_center = uint32_t(vertices.size());
	vertices.push_back({ glm::vec3(0.0f, -half, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f) });

	// Все треугольники обходятся против часовой стрелки при взгляде снаружи
	for (uint32_t i = 0; i < segments; ++i) {
		const uint32_t j = (i + 1) % segments;

		indices.insert(indices.end(), { segments + i, i, segments + j });
		indices.insert(indices.end(), { segments + j, i, j });

		indices.insert(indices.end(), { top_center, top_ring + j, top_ring + i });
		indices.insert(indices.end(), { bottom_center, bottom_ring + i, bottom_ring + j });
	}
}

void buildFloor(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, float half_size) {
	const glm::vec3 normal(0.0f, 1.0f, 0.0f);
	vertices = {
		{ glm::vec3(-half_size, 0.0f, -half_size), normal },
		{ glm::vec3(-half_size, 0.0f,  half_size), normal },
		{ glm::vec3( half_size, 0.0f,  half_size), normal },
		{ glm::vec3( half_size, 0.0f, -half_size), normal },
	};
	indices = { 0, 1, 2, 0, 2, 3 };
}

// ---------------------------------------------------------------------------
// Математика

glm::vec3 cameraForward(const Camera& c) {
	const float yaw = glm::radians(c.yaw);
	const float pitch = glm::radians(c.pitch);
	return glm::normalize(glm::vec3(
		std::cos(pitch) * std::sin(yaw),
		std::sin(pitch),
		-std::cos(pitch) * std::cos(yaw)
	));
}

glm::mat4 viewMatrix(const Camera& c) {
	return glm::lookAt(c.position, c.position + cameraForward(c), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 projectionMatrix(const Camera& c, float aspect) {
	glm::mat4 projection = glm::perspective(glm::radians(c.fov), aspect,
	                                        Camera::near_plane, Camera::far_plane);
	// GLM рассчитан на OpenGL (Y в клип-пространстве вверх), в Vulkan — вниз
	projection[1][1] *= -1.0f;
	return projection;
}

glm::mat4 normalMatrix(const glm::mat4& model) {
	return glm::mat4(glm::transpose(glm::inverse(glm::mat3(model))));
}

// ---------------------------------------------------------------------------
// Объекты Vulkan

bool createDescriptorLayouts() {
	auto& context = graphics::internal::context;

	// set 0: сцена — uniform-буфер и два storage-буфера с источниками
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout, nullptr,
	                                &scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create the scene descriptor set layout\n";
		return false;
	}

	// set 1: модель — матрица и материал одного объекта
	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &model_layout, nullptr,
	                                &model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create the model descriptor set layout\n";
		return false;
	}

	return true;
}

void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                           const graphics::Buffer& buffer) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer.buffer,
		.offset = 0,
		.range = buffer.size,
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

bool allocateSet(VkDescriptorSetLayout layout, VkDescriptorSet& out) {
	const VkDescriptorSetAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &layout,
	};
	return vkAllocateDescriptorSets(graphics::internal::context.device, &allocate, &out) == VK_SUCCESS;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	constexpr uint32_t model_count = 2;

	// maxSets и descriptorCount должны покрывать всё, что выделяется
	// из этого пула: набор сцены плюс по набору на модель
	const VkDescriptorPoolSize pool_sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 + model_count },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
	};

	const VkDescriptorPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + model_count,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create the descriptor pool\n";
		return false;
	}

	if (!graphics::createBuffer(scene_uniform_buffer, sizeof(SceneUniforms),
	                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
	    !graphics::createBuffer(point_light_buffer, sizeof(PointLightGPU) * light_capacity,
	                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
	    !graphics::createBuffer(spot_light_buffer, sizeof(SpotLightGPU) * light_capacity,
	                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
		return false;
	}

	if (!allocateSet(scene_set_layout, scene_descriptor_set)) {
		std::cerr << "Failed to allocate the scene descriptor set\n";
		return false;
	}

	writeBufferDescriptor(scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	                      scene_uniform_buffer);
	writeBufferDescriptor(scene_descriptor_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                      point_light_buffer);
	writeBufferDescriptor(scene_descriptor_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                      spot_light_buffer);

	for (Model* model : { &cylinder, &floor_plane }) {
		if (!graphics::createBuffer(model->uniform_buffer, sizeof(ModelUniforms),
		                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
			return false;
		}
		if (!allocateSet(model_set_layout, model->descriptor_set)) {
			std::cerr << "Failed to allocate a model descriptor set\n";
			return false;
		}
		writeBufferDescriptor(model->descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		                      model->uniform_buffer);
	}

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	// Layout конвейера перечисляет оба набора в порядке, который ждёт шейдер
	const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, model_set_layout };

	const VkPipelineLayoutCreateInfo layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &layout, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create the pipeline layout\n";
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
				.offset = offsetof(Vertex, normal),
			},
		},
		.layout = pipeline_layout,
		.render_pass = context.render_pass,
	};

	pipeline = graphics::createGraphicsPipeline(desc);
	return pipeline != VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Управление камерой

void updateCamera(float dt) {
	ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureMouse && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
		constexpr float sensitivity = 0.15f;
		camera.yaw += io.MouseDelta.x * sensitivity;
		camera.pitch = glm::clamp(camera.pitch - io.MouseDelta.y * sensitivity, -89.0f, 89.0f);
	}

	if (io.WantCaptureKeyboard) {
		return;
	}

	const glm::vec3 forward = cameraForward(camera);
	const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));

	glm::vec3 motion(0.0f);
	if (ImGui::IsKeyDown(ImGuiKey_W)) { motion += forward; }
	if (ImGui::IsKeyDown(ImGuiKey_S)) { motion -= forward; }
	if (ImGui::IsKeyDown(ImGuiKey_D)) { motion += right; }
	if (ImGui::IsKeyDown(ImGuiKey_A)) { motion -= right; }
	if (ImGui::IsKeyDown(ImGuiKey_E)) { motion += glm::vec3(0.0f, 1.0f, 0.0f); }
	if (ImGui::IsKeyDown(ImGuiKey_Q)) { motion -= glm::vec3(0.0f, 1.0f, 0.0f); }

	if (glm::dot(motion, motion) > 0.0f) {
		camera.position += glm::normalize(motion) * camera.move_speed * dt;
	}
}

// ---------------------------------------------------------------------------
// Интерфейс

void drawUI() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(400.0f, 610.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 2 - Camera and lighting");

	ImGui::Text("%.1f FPS  |  %u point, %u spot lights",
	            double(ImGui::GetIO().Framerate),
	            uint32_t(point_lights.size()), uint32_t(spot_lights.size()));

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position", &camera.position.x, 0.05f);
		ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f);
		ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f);
		ImGui::SliderFloat("FOV", &camera.fov, 20.0f, 120.0f);
	}

	if (ImGui::CollapsingHeader("Ambient and sun", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Ambient color", &ambient_color.x);
		ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 1.0f);
		ImGui::Spacing();
		ImGui::DragFloat3("Sun direction", &sun.direction.x, 0.02f);
		ImGui::ColorEdit3("Sun color", &sun.color.x);
		ImGui::SliderFloat("Sun intensity", &sun.intensity, 0.0f, 3.0f);
	}

	if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Albedo", &cylinder_material.albedo.x);
		ImGui::ColorEdit3("Specular", &cylinder_material.specular.x);
		ImGui::SliderFloat("Shininess", &cylinder_material.shininess, 1.0f, 256.0f);
	}

	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::BeginDisabled(point_lights.size() >= light_capacity);
		if (ImGui::Button("Add point light")) {
			point_lights.push_back(PointLight{});
		}
		ImGui::EndDisabled();

		for (size_t i = 0; i < point_lights.size();) {
			ImGui::PushID(int(i));
			bool remove = false;
			if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Point light %zu", i)) {
				ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.05f);
				ImGui::ColorEdit3("Color", &point_lights[i].color.x);
				ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 80.0f);
				remove = ImGui::Button("Remove");
				ImGui::TreePop();
			}
			ImGui::PopID();

			if (remove) {
				point_lights.erase(point_lights.begin() + long(i));
			} else {
				++i;
			}
		}
	}

	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::BeginDisabled(spot_lights.size() >= light_capacity);
		if (ImGui::Button("Add spot light")) {
			spot_lights.push_back(SpotLight{});
		}
		ImGui::EndDisabled();

		for (size_t i = 0; i < spot_lights.size();) {
			ImGui::PushID(int(1000 + i));
			bool remove = false;
			if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Spot light %zu", i)) {
				ImGui::DragFloat3("Position", &spot_lights[i].position.x, 0.05f);
				ImGui::DragFloat3("Direction", &spot_lights[i].direction.x, 0.02f);
				ImGui::ColorEdit3("Color", &spot_lights[i].color.x);
				ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.0f, 200.0f);
				ImGui::SliderFloat("Inner angle", &spot_lights[i].inner_angle, 1.0f, 80.0f);
				ImGui::SliderFloat("Outer angle", &spot_lights[i].outer_angle, 1.0f, 80.0f);
				// Яркая середина не должна быть шире конуса затухания
				spot_lights[i].inner_angle = glm::min(spot_lights[i].inner_angle,
				                                      spot_lights[i].outer_angle);
				remove = ImGui::Button("Remove");
				ImGui::TreePop();
			}
			ImGui::PopID();

			if (remove) {
				spot_lights.erase(spot_lights.begin() + long(i));
			} else {
				++i;
			}
		}
	}

	ImGui::End();
}

// ---------------------------------------------------------------------------
// Загрузка данных кадра

void uploadSceneData(float aspect) {
	const glm::mat4 view_projection = projectionMatrix(camera, aspect) * viewMatrix(camera);

	const SceneUniforms uniforms = {
		.view_projection = view_projection,
		.camera_position = glm::vec4(camera.position, 1.0f),
		.ambient = glm::vec4(ambient_color, ambient_intensity),
		.sun_direction = glm::vec4(sun.direction, 0.0f),
		.sun_color = glm::vec4(sun.color, sun.intensity),
		.light_counts = glm::uvec4(uint32_t(point_lights.size()),
		                           uint32_t(spot_lights.size()), 0, 0),
	};
	graphics::uploadBuffer(scene_uniform_buffer, &uniforms, sizeof(uniforms));

	if (!point_lights.empty()) {
		std::vector<PointLightGPU> packed(point_lights.size());
		for (size_t i = 0; i < point_lights.size(); ++i) {
			packed[i].position = glm::vec4(point_lights[i].position, 1.0f);
			packed[i].color = glm::vec4(point_lights[i].color, point_lights[i].intensity);
		}
		graphics::uploadBuffer(point_light_buffer, packed.data(),
		                       sizeof(PointLightGPU) * packed.size());
	}

	if (!spot_lights.empty()) {
		std::vector<SpotLightGPU> packed(spot_lights.size());
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			const SpotLight& light = spot_lights[i];
			// Косинусы считаются на CPU: шейдер сравнивает их напрямую,
			// не вызывая acos для каждого фрагмента
			packed[i].position = glm::vec4(light.position, std::cos(glm::radians(light.inner_angle)));
			packed[i].direction = glm::vec4(glm::normalize(light.direction),
			                                std::cos(glm::radians(light.outer_angle)));
			packed[i].color = glm::vec4(light.color, light.intensity);
		}
		graphics::uploadBuffer(spot_light_buffer, packed.data(),
		                       sizeof(SpotLightGPU) * packed.size());
	}
}

void uploadModelData(Model& model, const glm::mat4& transform, const Material& material) {
	const ModelUniforms uniforms = {
		.model = transform,
		.normal_matrix = normalMatrix(transform),
		.albedo = glm::vec4(material.albedo, 1.0f),
		.specular = glm::vec4(material.specular, material.shininess),
	};
	graphics::uploadBuffer(model.uniform_buffer, &uniforms, sizeof(uniforms));
}

void drawModel(VkCommandBuffer cmd, const Model& model) {
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &model.mesh.vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cmd, model.mesh.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	// set 0 держится весь кадр, от объекта к объекту меняется только set 1
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        1, 1, &model.descriptor_set, 0, nullptr);
	vkCmdDrawIndexed(cmd, model.mesh.index_count, 1, 0, 0, 0);
}

} // namespace

// ---------------------------------------------------------------------------

bool initialize() {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	buildCylinder(vertices, indices, cylinder_segments, cylinder_radius, cylinder_height);
	if (!graphics::createMesh(cylinder.mesh, vertices, indices)) {
		return false;
	}

	buildFloor(vertices, indices, floor_half_size);
	if (!graphics::createMesh(floor_plane.mesh, vertices, indices)) {
		return false;
	}

	return createDescriptorLayouts() && createDescriptors() && createPipeline();
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	for (Model* model : { &cylinder, &floor_plane }) {
		graphics::destroyBuffer(model->uniform_buffer);
		graphics::destroyMesh(model->mesh);
	}

	graphics::destroyBuffer(spot_light_buffer);
	graphics::destroyBuffer(point_light_buffer);
	graphics::destroyBuffer(scene_uniform_buffer);

	// Наборы дескрипторов освобождаются вместе с пулом
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);
}

void update(double time) {
	const float dt = last_time > 0.0 ? float(time - last_time) : 0.0f;
	last_time = time;

	updateCamera(dt);
	drawUI();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	auto& context = graphics::internal::context;
	const VkExtent2D extent = context.swapchain_extent;

	// prepare дождался fence, значит GPU с этими буферами уже закончил
	uploadSceneData(float(extent.width) / float(extent.height));
	uploadModelData(cylinder, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, cylinder_level, 0.0f)),
	                cylinder_material);
	uploadModelData(floor_plane, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, floor_level, 0.0f)),
	                floor_material);

	VkCommandBuffer cmd = fd.command_buffer;

	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin);

	const VkClearValue clear_values[] = {
		{ .color = { { 0.03f, 0.03f, 0.05f, 1.0f } } },
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

	// Данные сцены одинаковы для всех объектов, поэтому привязываются раз в кадр
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &scene_descriptor_set, 0, nullptr);

	drawModel(cmd, floor_plane);
	drawModel(cmd, cylinder);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
