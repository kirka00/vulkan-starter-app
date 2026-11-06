#include "application.hpp"
#include "graphics.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include <imgui.h>

#include <stb_image.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

namespace application {

namespace {


struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};

struct alignas(16) SceneUniforms {
	glm::mat4 view_projection;
	glm::vec4 camera_position;
	glm::vec4 ambient;        // rgb = цвет, a = интенсивность
	glm::vec4 sun_direction;  // xyz = направление, куда летит свет
	glm::vec4 sun_color;      // rgb = цвет, a = интенсивность
	glm::uvec4 light_counts;  // x = точечные источники, y = прожекторные
};

// Один элемент динамического uniform-буфера, элементы идут с шагом model_uniforms_stride
struct alignas(16) ModelUniforms {
	glm::mat4 model;
	glm::mat4 normal_matrix;
	glm::vec4 specular; // rgb = цвет блика, a = коэффициент блеска
	glm::vec4 emissive; // rgb = цвет свечения, a = интенсивность
};

// uint32_t, а не bool: в GLSL bool занимает 4 байта, а размер push-констант кратен 4
struct PushConstants {
	uint32_t flags;
	float time;
};

constexpr uint32_t flag_animated = 1u << 0;

struct alignas(16) PointLightGPU {
	glm::vec4 position;
	glm::vec4 color;
};

struct alignas(16) SpotLightGPU {
	glm::vec4 position;  // w = косинус внутреннего угла
	glm::vec4 direction; // w = косинус внешнего угла
	glm::vec4 color;
};


struct Camera {
	glm::vec3 position = glm::vec3(3.0f, 1.6f, 7.5f);
	float yaw = -20.0f;  // Градусы, поворот вокруг Y
	float pitch = -8.0f; // Градусы
	float fov = 60.0f;

	static constexpr float move_speed = 5.0f;
	static constexpr float near_plane = 0.1f;
	static constexpr float far_plane = 100.0f;
};

struct DirectionalLight {
	glm::vec3 direction = glm::vec3(-0.55f, -0.75f, -0.45f);
	glm::vec3 color = glm::vec3(1.0f, 0.96f, 0.88f);
	float intensity = 0.6f;
};

struct PointLight {
	glm::vec3 position = glm::vec3(3.0f, 0.4f, 2.2f);
	glm::vec3 color = glm::vec3(1.0f, 0.55f, 0.25f);
	float intensity = 10.0f;
};

struct SpotLight {
	glm::vec3 position = glm::vec3(-1.6f, 4.5f, 1.2f);
	glm::vec3 direction = glm::vec3(0.0f, -1.0f, -0.3f);
	glm::vec3 color = glm::vec3(0.55f, 0.8f, 1.0f);
	float intensity = 45.0f;
	float inner_angle = 14.0f; // Градусы, половина угла яркой середины
	float outer_angle = 24.0f; // Градусы, половина угла, где свет гаснет
};

// Пути к файлам текстур материала, nullptr — карты нет
struct MaterialDesc {
	const char* name;
	const char* albedo;
	const char* specular;
	const char* emissive;
};

const MaterialDesc material_descs[] = {
	{ "Bricks", "textures/bricks_albedo.jpg", "textures/bricks_specular.jpg", nullptr },
	{ "Wood", "textures/wood_albedo.jpg", "textures/wood_specular.jpg", nullptr },
	{ "Metal plate", "textures/metal_albedo.jpg", "textures/metal_specular.jpg", nullptr },
	{ "Stone tiles", "textures/stone_albedo.jpg", "textures/stone_specular.jpg", nullptr },
	{ "Lava", "textures/lava_albedo.jpg", "textures/lava_specular.jpg", "textures/lava_emissive.jpg" },
	{ "UV grid", "textures/uv_grid.png", nullptr, nullptr },
};

constexpr uint32_t material_count = sizeof(material_descs) / sizeof(material_descs[0]);

// Индексы в массиве textures; 0 — служебная текстура из одного чёрного пикселя
struct Material {
	uint32_t albedo = 0;
	uint32_t specular = 0;
	uint32_t emissive = 0;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct Model {
	const char* name;
	const graphics::Mesh* mesh;
	glm::mat4 transform;
	int material;
	glm::vec3 specular = glm::vec3(1.0f);
	float shininess = 32.0f;
	glm::vec3 emissive = glm::vec3(1.0f);
	float emissive_intensity = 1.5f;
	bool animated = false;
};

enum Filtering : int {
	filtering_nearest,
	filtering_bilinear,
	filtering_trilinear,
	filtering_count,
};

const char* const filtering_names[filtering_count] = {
	"Nearest, no mipmaps",
	"Bilinear, no mipmaps",
	"Trilinear (mipmaps)",
};

constexpr uint32_t cylinder_segments = 50;
constexpr float cylinder_radius = 1.0f;
constexpr float cylinder_height = 2.0f;

constexpr float room_half_size = 7.0f;
constexpr float wall_half_height = 3.0f;
constexpr float floor_level = -1.5f;
constexpr float cylinder_level = floor_level + cylinder_height * 0.5f;

constexpr uint32_t light_capacity = 16;
constexpr uint32_t max_models = 16;

// Состояние

graphics::Mesh cylinder_mesh;
graphics::Mesh floor_mesh;
graphics::Mesh wall_mesh;

std::vector<graphics::Texture> textures;
Material materials[material_count];
std::vector<Model> models;

VkSampler samplers[filtering_count];
int filtering = filtering_trilinear;
bool filtering_changed = false;

VkDescriptorSetLayout scene_set_layout;
VkDescriptorSetLayout material_set_layout;
VkDescriptorPool descriptor_pool;

VkDescriptorSet scene_descriptor_set;
graphics::Buffer scene_uniform_buffer;
graphics::Buffer point_light_buffer;
graphics::Buffer spot_light_buffer;

graphics::Buffer model_uniform_buffer;
VkDeviceSize model_uniforms_stride = 0;
VkDeviceSize uniform_alignment = 0;

VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

Camera camera;
DirectionalLight sun;
glm::vec3 ambient_color = glm::vec3(1.0f);
float ambient_intensity = 0.08f;
bool night = false;

std::vector<PointLight> point_lights = { PointLight{} };
std::vector<SpotLight> spot_lights = { SpotLight{} };

double last_time = 0.0;
float animation_time = 0.0f;
float animation_speed = 1.0f;

// Геометрия

void buildCylinder(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                   uint32_t segments, float radius, float height) {
	const float half = height * 0.5f;
	// Сколько раз текстура обернётся вокруг стенки, чтобы тексели не растягивались по окружности
	constexpr float wraps = 3.0f;

	vertices.clear();
	indices.clear();

	// Стенка: segments + 1 вершин в кольце, первая и последняя совпадают по позиции,
	// но у первой u = 0, у последней u = wraps — это шов текстуры
	// [0, N + 1) — верх стенки, [N + 1, 2N + 2) — низ стенки
	for (uint32_t row = 0; row < 2; ++row) {
		const float y = row == 0 ? half : -half;
		for (uint32_t i = 0; i <= segments; ++i) {
			const float s = float(i) / float(segments);
			const float angle = glm::two_pi<float>() * s;
			const glm::vec3 normal(std::cos(angle), 0.0f, std::sin(angle));
			vertices.push_back({ glm::vec3(radius * normal.x, y, radius * normal.z), normal,
			                     glm::vec2(wraps * s, float(row)) });
		}
	}

	// Крышки: развёртка — проекция круга на квадрат текстуры, u и v от 0 до 1
	auto cap = [&](float y, float ny) {
		const uint32_t ring = uint32_t(vertices.size());
		for (uint32_t i = 0; i < segments; ++i) {
			const float angle = glm::two_pi<float>() * float(i) / float(segments);
			const float c = std::cos(angle);
			const float s = std::sin(angle);
			vertices.push_back({ glm::vec3(radius * c, y, radius * s), glm::vec3(0.0f, ny, 0.0f),
			                     glm::vec2(0.5f + 0.5f * c, 0.5f + 0.5f * s) });
		}
		vertices.push_back({ glm::vec3(0.0f, y, 0.0f), glm::vec3(0.0f, ny, 0.0f), glm::vec2(0.5f) });
		return ring;
	};
	const uint32_t top_ring = cap(half, 1.0f);
	const uint32_t top_center = top_ring + segments;
	const uint32_t bottom_ring = cap(-half, -1.0f);
	const uint32_t bottom_center = bottom_ring + segments;

	// Все треугольники обходятся против часовой стрелки при взгляде снаружи
	const uint32_t bottom_row = segments + 1;
	for (uint32_t i = 0; i < segments; ++i) {
		const uint32_t j = (i + 1) % segments;

		indices.insert(indices.end(), { bottom_row + i, i, bottom_row + i + 1 });
		indices.insert(indices.end(), { bottom_row + i + 1, i, i + 1 });

		indices.insert(indices.end(), { top_center, top_ring + j, top_ring + i });
		indices.insert(indices.end(), { bottom_center, bottom_ring + i, bottom_ring + j });
	}
}

// Прямоугольник в плоскости XZ с нормалью +Y; tile — сколько единиц сцены
// занимает одно повторение текстуры
void buildPlane(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                float half_x, float half_z, float tile) {
	const glm::vec3 normal(0.0f, 1.0f, 0.0f);
	const float u = 2.0f * half_x / tile;
	const float v = 2.0f * half_z / tile;
	vertices = {
		{ glm::vec3(-half_x, 0.0f, -half_z), normal, glm::vec2(0.0f, 0.0f) },
		{ glm::vec3(-half_x, 0.0f,  half_z), normal, glm::vec2(0.0f, v) },
		{ glm::vec3( half_x, 0.0f,  half_z), normal, glm::vec2(u, v) },
		{ glm::vec3( half_x, 0.0f, -half_z), normal, glm::vec2(u, 0.0f) },
	};
	indices = { 0, 1, 2, 0, 2, 3 };
}

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
	projection[1][1] *= -1.0f;
	return projection;
}

glm::mat4 normalMatrix(const glm::mat4& model) {
	return glm::mat4(glm::transpose(glm::inverse(glm::mat3(model))));
}

// Загрузка ресурсов в VRAM

struct Image {
	stbi_uc* pixels = nullptr;
	uint32_t width = 0;
	uint32_t height = 0;
	VkFormat format = VK_FORMAT_UNDEFINED;
};

bool loadImage(const char* path, VkFormat format, Image& out) {
	int width = 0, height = 0, channels = 0;
	// Всегда 4 канала: форматы с тремя каналами GPU поддерживает плохо
	out.pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	if (out.pixels == nullptr) {
		std::cerr << "Failed to load image " << path << ": " << stbi_failure_reason()
		          << " (is the working directory the project root?)\n";
		return false;
	}
	out.width = uint32_t(width);
	out.height = uint32_t(height);
	out.format = format;
	return true;
}

bool uploadResources() {
	// Цветные карты хранят цвет в sRGB, выборка переводит его в линейное пространство;
	// specular — это маска, её значения линейные
	std::vector<Image> images;
	auto add = [&](const char* path, VkFormat format) -> uint32_t {
		if (path == nullptr) {
			return 0;
		}
		Image image;
		if (!loadImage(path, format, image)) {
			return UINT32_MAX;
		}
		images.push_back(image);
		return uint32_t(images.size()); // +1 за чёрный пиксель в начале textures
	};

	bool ok = true;
	for (uint32_t i = 0; i < material_count; ++i) {
		materials[i].albedo = add(material_descs[i].albedo, VK_FORMAT_R8G8B8A8_SRGB);
		materials[i].specular = add(material_descs[i].specular, VK_FORMAT_R8G8B8A8_UNORM);
		materials[i].emissive = add(material_descs[i].emissive, VK_FORMAT_R8G8B8A8_SRGB);
		ok = ok && materials[i].albedo != UINT32_MAX && materials[i].albedo != 0 &&
		     materials[i].specular != UINT32_MAX && materials[i].emissive != UINT32_MAX;
	}

	std::vector<Vertex> cylinder_vertices, floor_vertices, wall_vertices;
	std::vector<uint32_t> cylinder_indices, floor_indices, wall_indices;
	buildCylinder(cylinder_vertices, cylinder_indices, cylinder_segments, cylinder_radius, cylinder_height);
	buildPlane(floor_vertices, floor_indices, room_half_size, room_half_size, 2.5f);
	buildPlane(wall_vertices, wall_indices, room_half_size, wall_half_height, 2.0f);

	// Размер staging-буфера — сумма всего, что поедет в VRAM
	const uint32_t black_pixel = 0xff000000;
	VkDeviceSize capacity = graphics::uploadFootprint(sizeof(black_pixel));
	for (const Image& image : images) {
		capacity += graphics::uploadFootprint(VkDeviceSize(image.width) * image.height * 4);
	}
	for (const auto* v : { &cylinder_vertices, &floor_vertices, &wall_vertices }) {
		capacity += graphics::uploadFootprint(sizeof(Vertex) * v->size());
	}
	for (const auto* i : { &cylinder_indices, &floor_indices, &wall_indices }) {
		capacity += graphics::uploadFootprint(sizeof(uint32_t) * i->size());
	}

	graphics::Uploader uploader;
	ok = ok && graphics::beginUpload(uploader, capacity);

	textures.resize(images.size() + 1);
	ok = ok && graphics::uploadToTexture(uploader, textures[0], &black_pixel, 1, 1,
	                                     VK_FORMAT_R8G8B8A8_UNORM);
	for (size_t i = 0; ok && i < images.size(); ++i) {
		ok = graphics::uploadToTexture(uploader, textures[i + 1], images[i].pixels,
		                               images[i].width, images[i].height, images[i].format);
	}

	ok = ok && graphics::createMesh(uploader, cylinder_mesh, cylinder_vertices, cylinder_indices) &&
	     graphics::createMesh(uploader, floor_mesh, floor_vertices, floor_indices) &&
	     graphics::createMesh(uploader, wall_mesh, wall_vertices, wall_indices);

	if (uploader.command_buffer != VK_NULL_HANDLE) {
		ok = graphics::endUpload(uploader) && ok;
	}

	// Пиксели уже скопированы в staging-буфер и на GPU, копия в RAM больше не нужна
	for (Image& image : images) {
		stbi_image_free(image.pixels);
	}

	return ok;
}

// Объекты Vulkan

bool createSamplers() {
	auto& context = graphics::internal::context;

	for (int i = 0; i < filtering_count; ++i) {
		const bool nearest = i == filtering_nearest;
		const bool mipmaps = i == filtering_trilinear;

		const VkSamplerCreateInfo sampler = {
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
			.minFilter = nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
			.mipmapMode = mipmaps ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
			// Координаты за пределами [0, 1] повторяют текстуру — на этом держится тайлинг пола и стен
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.anisotropyEnable = VK_FALSE,
			.minLod = 0.0f,
			// maxLod = 0 запрещает уровни меньше исходного изображения
			.maxLod = mipmaps ? VK_LOD_CLAMP_NONE : 0.0f,
		};

		if (vkCreateSampler(context.device, &sampler, nullptr, &samplers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create a sampler\n";
			return false;
		}
	}
	return true;
}

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

	// set 1: материал — общий динамический uniform-буфер моделей и три текстуры материала
	const VkDescriptorSetLayoutBinding material_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo material_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(material_bindings) / sizeof(material_bindings[0]),
		.pBindings = material_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &material_layout, nullptr,
	                                &material_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create the material descriptor set layout\n";
		return false;
	}

	return true;
}

void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                           const graphics::Buffer& buffer, VkDeviceSize range) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer.buffer,
		.offset = 0,
		.range = range,
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

void writeMaterialImages(const Material& material, VkSampler sampler) {
	const uint32_t maps[] = { material.albedo, material.specular, material.emissive };

	VkDescriptorImageInfo image_infos[3];
	VkWriteDescriptorSet writes[3];
	for (uint32_t i = 0; i < 3; ++i) {
		image_infos[i] = {
			.sampler = sampler,
			.imageView = textures[maps[i]].view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};
		writes[i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = material.descriptor_set,
			.dstBinding = 1 + i,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &image_infos[i],
		};
	}

	vkUpdateDescriptorSets(graphics::internal::context.device, 3, writes, 0, nullptr);
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

	const VkDescriptorPoolSize pool_sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, material_count },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * material_count },
	};

	const VkDescriptorPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + material_count,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create the descriptor pool\n";
		return false;
	}

	// Сдвиг динамического буфера обязан быть кратен minUniformBufferOffsetAlignment,
	// поэтому шаг между элементами — размер структуры, округлённый вверх до этого числа
	VkPhysicalDeviceProperties props = {};
	vkGetPhysicalDeviceProperties(context.physical_device, &props);
	uniform_alignment = props.limits.minUniformBufferOffsetAlignment;
	model_uniforms_stride = (sizeof(ModelUniforms) + uniform_alignment - 1) & ~(uniform_alignment - 1);

	if (!graphics::createBuffer(scene_uniform_buffer, sizeof(SceneUniforms),
	                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
	    !graphics::createBuffer(point_light_buffer, sizeof(PointLightGPU) * light_capacity,
	                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
	    !graphics::createBuffer(spot_light_buffer, sizeof(SpotLightGPU) * light_capacity,
	                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
	    !graphics::createBuffer(model_uniform_buffer, model_uniforms_stride * max_models,
	                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
		return false;
	}

	if (!allocateSet(scene_set_layout, scene_descriptor_set)) {
		std::cerr << "Failed to allocate the scene descriptor set\n";
		return false;
	}

	writeBufferDescriptor(scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	                      scene_uniform_buffer, scene_uniform_buffer.size);
	writeBufferDescriptor(scene_descriptor_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                      point_light_buffer, point_light_buffer.size);
	writeBufferDescriptor(scene_descriptor_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                      spot_light_buffer, spot_light_buffer.size);

	for (Material& material : materials) {
		if (!allocateSet(material_set_layout, material.descriptor_set)) {
			std::cerr << "Failed to allocate a material descriptor set\n";
			return false;
		}
		// range — один элемент массива, а не весь буфер: к нему прибавится динамический сдвиг,
		// и сдвиг + range не должен выходить за конец буфера
		writeBufferDescriptor(material.descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
		                      model_uniform_buffer, sizeof(ModelUniforms));
		writeMaterialImages(material, samplers[filtering]);
	}

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, material_set_layout };

	const VkPushConstantRange push_range = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};

	const VkPipelineLayoutCreateInfo layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
	};

	if (vkCreatePipelineLayout(context.device, &layout, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create the pipeline layout\n";
		return false;
	}

	const graphics::GraphicsPipelineDesc desc = {
		.vertex_shader_path = "shaders/scene.vert.spv",
		.fragment_shader_path = "shaders/scene.frag.spv",
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
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		},
		.layout = pipeline_layout,
		.render_pass = context.render_pass,
	};

	pipeline = graphics::createGraphicsPipeline(desc);
	return pipeline != VK_NULL_HANDLE;
}

void createScene() {
	const glm::mat4 identity(1.0f);
	const glm::mat4 upright = glm::rotate(identity, glm::half_pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
	const float wall_level = floor_level + wall_half_height;

	models = {
		{ .name = "Floor", .mesh = &floor_mesh,
		  .transform = glm::translate(identity, glm::vec3(0.0f, floor_level, 0.0f)),
		  .material = 1, .specular = glm::vec3(0.6f), .shininess = 24.0f },
		// Плоскость поднята в вертикаль: нормаль +Y переходит в +Z, а затем в +X
		{ .name = "Back wall", .mesh = &wall_mesh,
		  .transform = glm::translate(identity, glm::vec3(0.0f, wall_level, -room_half_size)) * upright,
		  .material = 0, .specular = glm::vec3(0.5f), .shininess = 16.0f },
		{ .name = "Left wall", .mesh = &wall_mesh,
		  .transform = glm::translate(identity, glm::vec3(-room_half_size, wall_level, 0.0f)) *
		               glm::rotate(identity, glm::half_pi<float>(), glm::vec3(0.0f, 1.0f, 0.0f)) * upright,
		  .material = 0, .specular = glm::vec3(0.5f), .shininess = 16.0f },
		{ .name = "Cylinder", .mesh = &cylinder_mesh,
		  .transform = glm::translate(identity, glm::vec3(-1.6f, cylinder_level, 0.0f)),
		  .material = 2, .specular = glm::vec3(1.0f), .shininess = 64.0f },
		{ .name = "Lava cylinder", .mesh = &cylinder_mesh,
		  .transform = glm::translate(identity, glm::vec3(1.6f, cylinder_level, 0.0f)),
		  .material = 4, .specular = glm::vec3(0.8f), .shininess = 32.0f,
		  .emissive = glm::vec3(1.0f), .emissive_intensity = 2.0f, .animated = true },
	};
}

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

// Интерфейс

void drawModelUI(Model& model) {
	if (ImGui::BeginCombo("Material", material_descs[model.material].name)) {
		for (int m = 0; m < int(material_count); ++m) {
			if (ImGui::Selectable(material_descs[m].name, model.material == m)) {
				model.material = m;
			}
		}
		ImGui::EndCombo();
	}
	ImGui::ColorEdit3("Specular", &model.specular.x);
	ImGui::SliderFloat("Shininess", &model.shininess, 1.0f, 256.0f);
	ImGui::ColorEdit3("Emissive", &model.emissive.x);
	ImGui::SliderFloat("Emissive intensity", &model.emissive_intensity, 0.0f, 5.0f);
	ImGui::Checkbox("Animated sampling", &model.animated);
}

void drawUI() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(420.0f, 520.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 3 - Texture mapping");

	if (ImGui::CollapsingHeader("Camera")) {
		ImGui::DragFloat3("Position", &camera.position.x, 0.05f);
		ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f);
		ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f);
		ImGui::SliderFloat("FOV", &camera.fov, 20.0f, 120.0f);
	}

	if (ImGui::CollapsingHeader("Ambient and sun", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Night", &night);
		ImGui::ColorEdit3("Ambient color", &ambient_color.x);
		ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 1.0f);
		ImGui::Spacing();
		ImGui::DragFloat3("Sun direction", &sun.direction.x, 0.02f);
		ImGui::ColorEdit3("Sun color", &sun.color.x);
		ImGui::SliderFloat("Sun intensity", &sun.intensity, 0.0f, 3.0f);
	}

	if (ImGui::CollapsingHeader("Textures", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Combo("Filtering", &filtering, filtering_names, filtering_count)) {
			filtering_changed = true;
		}
		ImGui::SliderFloat("Animation speed", &animation_speed, 0.0f, 4.0f);
	}

	if (ImGui::CollapsingHeader("Models", ImGuiTreeNodeFlags_DefaultOpen)) {
		for (size_t i = 0; i < models.size(); ++i) {
			ImGui::PushID(int(i));
			if (ImGui::TreeNode(models[i].name)) {
				drawModelUI(models[i]);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}

	if (ImGui::CollapsingHeader("Point lights")) {
		ImGui::BeginDisabled(point_lights.size() >= light_capacity);
		if (ImGui::Button("Add point light")) {
			point_lights.push_back(PointLight{});
		}
		ImGui::EndDisabled();

		for (size_t i = 0; i < point_lights.size();) {
			ImGui::PushID(int(100 + i));
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

	if (ImGui::CollapsingHeader("Spot lights")) {
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


// Загрузка данных кадра

void uploadSceneData(float aspect) {
	const glm::mat4 view_projection = projectionMatrix(camera, aspect) * viewMatrix(camera);

	// Ночью остаётся только свечение emissive-текстур
	const uint32_t point_count = night ? 0 : uint32_t(point_lights.size());
	const uint32_t spot_count = night ? 0 : uint32_t(spot_lights.size());

	const SceneUniforms uniforms = {
		.view_projection = view_projection,
		.camera_position = glm::vec4(camera.position, 1.0f),
		.ambient = glm::vec4(ambient_color, night ? 0.0f : ambient_intensity),
		.sun_direction = glm::vec4(sun.direction, 0.0f),
		.sun_color = glm::vec4(sun.color, night ? 0.0f : sun.intensity),
		.light_counts = glm::uvec4(point_count, spot_count, 0, 0),
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
			packed[i].position = glm::vec4(light.position, std::cos(glm::radians(light.inner_angle)));
			packed[i].direction = glm::vec4(glm::normalize(light.direction),
			                                std::cos(glm::radians(light.outer_angle)));
			packed[i].color = glm::vec4(light.color, light.intensity);
		}
		graphics::uploadBuffer(spot_light_buffer, packed.data(),
		                       sizeof(SpotLightGPU) * packed.size());
	}
}

// Все модели пишутся в один буфер: i-я модель лежит по адресу начала + i * stride
void uploadModelData() {
	for (size_t i = 0; i < models.size(); ++i) {
		const Model& model = models[i];
		const ModelUniforms uniforms = {
			.model = model.transform,
			.normal_matrix = normalMatrix(model.transform),
			.specular = glm::vec4(model.specular, model.shininess),
			.emissive = glm::vec4(model.emissive, model.emissive_intensity),
		};
		graphics::uploadBuffer(model_uniform_buffer, &uniforms, sizeof(uniforms),
		                       VkDeviceSize(i) * model_uniforms_stride);
	}
}

void drawModel(VkCommandBuffer cmd, const Model& model, uint32_t index) {
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &model.mesh->vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cmd, model.mesh->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	// Набор выбирается по материалу, а элемент буфера моделей — динамическим сдвигом
	const uint32_t dynamic_offset = uint32_t(index * model_uniforms_stride);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        1, 1, &materials[model.material].descriptor_set, 1, &dynamic_offset);

	const PushConstants push = {
		.flags = model.animated ? flag_animated : 0u,
		.time = animation_time,
	};
	vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
	                   0, sizeof(push), &push);

	vkCmdDrawIndexed(cmd, model.mesh->index_count, 1, 0, 0, 0);
}

} // namespace

bool initialize() {
	static_assert(sizeof(PushConstants) % 4 == 0, "push constant size must be a multiple of 4");

	if (!uploadResources()) {
		return false;
	}

	createScene();
	if (models.size() > max_models) {
		std::cerr << "Too many models for the dynamic uniform buffer\n";
		return false;
	}

	return createSamplers() && createDescriptorLayouts() && createDescriptors() && createPipeline();
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	graphics::destroyBuffer(model_uniform_buffer);
	graphics::destroyBuffer(spot_light_buffer);
	graphics::destroyBuffer(point_light_buffer);
	graphics::destroyBuffer(scene_uniform_buffer);

	// Наборы дескрипторов освобождаются вместе с пулом
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, material_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);

	for (VkSampler sampler : samplers) {
		vkDestroySampler(context.device, sampler, nullptr);
	}
	for (graphics::Texture& texture : textures) {
		graphics::destroyTexture(texture);
	}
	for (graphics::Mesh* mesh : { &cylinder_mesh, &floor_mesh, &wall_mesh }) {
		graphics::destroyMesh(*mesh);
	}
}

void update(double time) {
	const float dt = last_time > 0.0 ? float(time - last_time) : 0.0f;
	last_time = time;
	animation_time += dt * animation_speed;

	updateCamera(dt);
	drawUI();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	auto& context = graphics::internal::context;
	const VkExtent2D extent = context.swapchain_extent;

	// prepare дождался fence, значит GPU с этими буферами и наборами уже закончил
	if (filtering_changed) {
		for (const Material& material : materials) {
			writeMaterialImages(material, samplers[filtering]);
		}
		filtering_changed = false;
	}

	uploadSceneData(float(extent.width) / float(extent.height));
	uploadModelData();

	VkCommandBuffer cmd = fd.command_buffer;

	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin);

	const VkClearValue clear_values[] = {
		{ .color = { { 0.01f, 0.01f, 0.015f, 1.0f } } },
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

	for (size_t i = 0; i < models.size(); ++i) {
		drawModel(cmd, models[i], uint32_t(i));
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
