#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_normal;

layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;        // rgb = цвет, a = интенсивность
	vec4 sun_direction;  // xyz = направление, куда летит свет
	vec4 sun_color;      // rgb = цвет, a = интенсивность
	uvec4 light_counts;  // x = точечные источники, y = прожекторные
} scene;

layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;  // transpose(inverse(model)), сохраняет нормали при неравномерном масштабе
	vec4 albedo;
	vec4 specular;       // rgb = цвет блика, a = коэффициент блеска
} model;

void main() {
	vec4 world_position = model.model * vec4(in_position, 1.0);

	// Освещение считается в мировом пространстве, поэтому передаём обе величины
	out_world_position = world_position.xyz;
	out_normal = mat3(model.normal_matrix) * in_normal;

	gl_Position = scene.view_projection * world_position;
}
