#version 450

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec4 out_color;

struct PointLight {
	vec4 position;  // xyz = позиция
	vec4 color;     // rgb = цвет, a = интенсивность
};

struct SpotLight {
	vec4 position;   // xyz = позиция, w = косинус внутреннего угла
	vec4 direction;  // xyz = направление, w = косинус внешнего угла
	vec4 color;      // rgb = цвет, a = интенсивность
};

layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 sun_direction;
	vec4 sun_color;
	uvec4 light_counts;
} scene;

layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
	PointLight lights[];
} point_lights;

layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
	SpotLight lights[];
} spot_lights;

layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
} model;

vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 radiance) {
	float n_dot_l = max(dot(n, l), 0.0);

	vec3 diffuse = model.albedo.rgb * n_dot_l;

	vec3 h = normalize(l + v);
	float highlight = n_dot_l > 0.0 ? pow(max(dot(n, h), 0.0), max(model.specular.a, 1.0)) : 0.0;
	vec3 specular = model.specular.rgb * highlight;

	return (diffuse + specular) * radiance;
}

void main() {
	vec3 n = normalize(in_normal);
	vec3 v = normalize(scene.camera_position.xyz - in_world_position);

	// Рассеянный свет приближает свет, приходящий со всего окружения
	vec3 color = scene.ambient.rgb * scene.ambient.a * model.albedo.rgb;

	// Направленный источник бесконечно далёк: ни позиции, ни затухания
	color += blinnPhong(n, v, normalize(-scene.sun_direction.xyz),
	                    scene.sun_color.rgb * scene.sun_color.a);

	for (uint i = 0; i < scene.light_counts.x; ++i) {
		PointLight light = point_lights.lights[i];

		vec3 to_light = light.position.xyz - in_world_position;
		float distance = max(length(to_light), 0.001);
		// Закон обратных квадратов
		float falloff = 1.0 / (distance * distance);

		color += blinnPhong(n, v, to_light / distance, light.color.rgb * light.color.a * falloff);
	}

	for (uint i = 0; i < scene.light_counts.y; ++i) {
		SpotLight light = spot_lights.lights[i];

		vec3 to_light = light.position.xyz - in_world_position;
		float distance = max(length(to_light), 0.001);
		vec3 l = to_light / distance;
		float falloff = 1.0 / (distance * distance);

		// Мягкий край: внутри внутреннего конуса полная яркость, снаружи внешнего
		// ноль, между ними плавный переход
		float cosine = dot(normalize(light.direction.xyz), -l);
		float cone = smoothstep(light.direction.w, light.position.w, cosine);

		color += blinnPhong(n, v, l, light.color.rgb * light.color.a * falloff * cone);
	}

	out_color = vec4(color, 1.0);
}
