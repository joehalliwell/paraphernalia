#version 330
#ifdef GL_ES
precision mediump float;
#endif

uniform float u_time;
uniform float u_bar;

out vec4 fragColor;

void main() {
	fragColor = vec4(abs(sin(u_time)),0.0,0.0,1.0);
}
