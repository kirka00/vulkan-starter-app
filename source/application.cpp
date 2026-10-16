#include "application.hpp"
#include "graphics.hpp"

#include <cmath>
#include <vector>

#include <imgui.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // Глубина в клип-пространстве Vulkan — [0, 1]
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace application {

namespace {

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

constexpr uint32_t cylinder_segments = 50;
constexpr float cylinder_radius = 1.0f;
constexpr float cylinder_height = 2.0f;

graphics::Mesh cylinder_mesh;

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

} // namespace

// ---------------------------------------------------------------------------

bool initialize() {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	buildCylinder(vertices, indices, cylinder_segments, cylinder_radius, cylinder_height);

	return graphics::createMesh(cylinder_mesh, vertices, indices);
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	graphics::destroyMesh(cylinder_mesh);
}

void update([[maybe_unused]] double time) {
	ImGui::ShowDemoWindow();
}

void render(const graphics::internal::FrameData& fd) {
	(void)fd;
}

} // namespace application
