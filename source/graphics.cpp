#include "graphics.hpp"

#include <fstream>
#include <iostream>

#include <vulkan/vulkan.h>

#include "graphics_internal.hpp"

namespace graphics {

bool createBuffer(Buffer& out, VkDeviceSize size, VkBufferUsageFlags usage,
                  const void* initial_data) {
	auto& context = internal::context;

	const VkBufferCreateInfo buffer = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo allocation_info = {};

	if (vmaCreateBuffer(context.allocator, &buffer, &allocation, &out.buffer,
	                    &out.allocation, &allocation_info) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer of " << size << " bytes\n";
		return false;
	}

	out.mapped = allocation_info.pMappedData;
	out.size = size;

	if (initial_data != nullptr) {
		uploadBuffer(out, initial_data, size);
	}

	return true;
}

void uploadBuffer(Buffer& buffer, const void* data, VkDeviceSize size) {
	std::memcpy(buffer.mapped, data, size_t(size));
	// На host-coherent памяти ничего не делает, на остальной — обязателен
	vmaFlushAllocation(internal::context.allocator, buffer.allocation, 0, size);
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(internal::context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

void destroyMesh(Mesh& mesh) {
	destroyBuffer(mesh.index_buffer);
	destroyBuffer(mesh.vertex_buffer);
	mesh = {};
}

VkShaderModule loadShaderModule(const std::string& path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file " << path
		          << " (is the working directory the project root?)\n";
		return VK_NULL_HANDLE;
	}

	const std::streamsize size = file.tellg();
	file.seekg(0);

	std::vector<uint32_t> code((size_t(size) + sizeof(uint32_t) - 1) / sizeof(uint32_t));
	file.read(reinterpret_cast<char*>(code.data()), size);

	const VkShaderModuleCreateInfo module = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size_t(size),
		.pCode = code.data(),
	};

	VkShaderModule result = VK_NULL_HANDLE;
	if (vkCreateShaderModule(internal::context.device, &module, nullptr, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from " << path << '\n';
		return VK_NULL_HANDLE;
	}

	return result;
}

VkPipeline createGraphicsPipeline(const GraphicsPipelineDesc& desc) {
	auto& context = internal::context;

	VkShaderModule vertex_shader = loadShaderModule(desc.vertex_shader_path);
	VkShaderModule fragment_shader = loadShaderModule(desc.fragment_shader_path);
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return VK_NULL_HANDLE;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = uint32_t(desc.vertex_bindings.size()),
		.pVertexBindingDescriptions = desc.vertex_bindings.data(),
		.vertexAttributeDescriptionCount = uint32_t(desc.vertex_attributes.size()),
		.pVertexAttributeDescriptions = desc.vertex_attributes.data(),
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = desc.topology,
	};

	// Сами viewport и scissor задаются через vkCmdSetViewport и vkCmdSetScissor
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = desc.polygon_mode,
		.cullMode = desc.cull_mode,
		.frontFace = desc.front_face,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = desc.depth_test,
		.depthWriteEnable = desc.depth_write,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = desc.layout,
		.renderPass = desc.render_pass,
		.subpass = 0,
	};

	VkPipeline result = VK_NULL_HANDLE;
	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline,
	                              nullptr, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		result = VK_NULL_HANDLE;
	}

	// Модули уже встроены в конвейер, их можно сразу освободить
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	return result;
}

} // namespace graphics
