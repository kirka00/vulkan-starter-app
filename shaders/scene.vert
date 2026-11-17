#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_uv;
layout(location = 3) out vec4 out_light_position;

layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;        // rgb = цвет, a = интенсивность
	vec4 sun_direction;  // xyz = направление, куда летит свет
	vec4 sun_color;      // rgb = цвет, a = интенсивность
	uvec4 light_counts;  // x = точечные источники, y = прожекторные
	mat4 light_view_projection;  // мировые координаты -> пространство отсечения солнца
	vec4 shadow_params;  // x = тени включены, y = PCF 3x3, z = 1 / размер карты теней
} scene;

// Динамический uniform-буфер: какой элемент массива читать, задаёт сдвиг в vkCmdBindDescriptorSets
layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec4 specular;       // rgb = цвет блика, a = коэффициент блеска
	vec4 emissive;       // rgb = цвет свечения, a = интенсивность
} model;

void main() {
	vec4 world_position = model.model * vec4(in_position, 1.0);

	out_world_position = world_position.xyz;
	out_normal = mat3(model.normal_matrix) * in_normal;
	out_uv = in_uv;
	out_light_position = scene.light_view_projection * world_position;

	gl_Position = scene.view_projection * world_position;
}
