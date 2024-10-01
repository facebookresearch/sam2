#version 300 es
// Copyright (c) Meta Platforms, Inc. and affiliates.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

precision mediump float;
precision mediump sampler3D;

in vec2 vTexCoord;

uniform sampler2D uSampler;
uniform sampler3D uColorGradeLUT;
uniform mediump vec2 uSize;

out vec4 fragColor;

void main() {

  // texel color
  vec3 color = texture(uSampler, vTexCoord).rgb;
  vec3 gradedColor = texture(uColorGradeLUT, color).rgb;
  fragColor = vec4(gradedColor, 1);
}