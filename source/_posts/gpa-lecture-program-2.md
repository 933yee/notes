---
title: Graphcis Programming and Application Lecture Program 解析 (2)
date: 2024-11-11 10:53:03
tags:
category:
math: true
---

# Grass

```cpp
#include ".../.../Include/Common.h"

using namespace glm;

static const char * grass_vs_source[] =
{
    "#version 410 core                                                                                           \n"
    "                                                                                                            \n"
    "in vec4 vVertex;                                                                                            \n"
    "                                                                                                            \n"
    "out vec4 color;                                                                                             \n"
    "                                                                                                            \n"
    "uniform mat4 mvpMatrix;                                                                                     \n"
    "                                                                                                            \n"
    "int random(int seed, int iterations)                                                                        \n"
    "{                                                                                                           \n"
    "    int value = seed;                                                                                       \n"
    "    int n;                                                                                                  \n"
    "                                                                                                            \n"
    "    for (n = 0; n < iterations; n++) {                                                                      \n"
    "        value = ((value >> 7) ^ (value << 9)) * 15485863;                                                   \n"
    "    }                                                                                                       \n"
    "                                                                                                            \n"
    "    return value;                                                                                           \n"
    "}                                                                                                           \n"
    "                                                                                                            \n"
    "vec4 random_vector(int seed)                                                                                \n"
    "{                                                                                                           \n"
    "    int r = random(gl_InstanceID, 4);                                                                       \n"
    "    int g = random(r, 2);                                                                                   \n"
    "    int b = random(g, 2);                                                                                   \n"
    "    int a = random(b, 2);                                                                                   \n"
    "                                                                                                            \n"
    "    return vec4(float(r & 0x3FF) / 1024.0,                                                                  \n"
    "                float(g & 0x3FF) / 1024.0,                                                                  \n"
    "                float(b & 0x3FF) / 1024.0,                                                                  \n"
    "                float(a & 0x3FF) / 1024.0);                                                                 \n"
    "}                                                                                                           \n"
    "                                                                                                            \n"
    "mat4 construct_rotation_matrix(float angle)                                                                 \n"
    "{                                                                                                           \n"
    "    float st = sin(angle);                                                                                  \n"
    "    float ct = cos(angle);                                                                                  \n"
    "                                                                                                            \n"
    "    return mat4(vec4(ct, 0.0, st, 0.0),                                                                     \n"
    "                vec4(0.0, 1.0, 0.0, 0.0),                                                                   \n"
    "                vec4(-st, 0.0, ct, 0.0),                                                                    \n"
    "                vec4(0.0, 0.0, 0.0, 1.0));                                                                  \n"
    "}                                                                                                           \n"
    "                                                                                                            \n"
    "void main(void)                                                                                             \n"
    "{                                                                                                           \n"
    "    vec4 offset = vec4(float(gl_InstanceID >> 10) - 512.0,                                                  \n"
    "                       0.0f,                                                                                \n"
    "                       float(gl_InstanceID & 0x3FF) - 512.0,                                                \n"
    "                       0.0f);                                                                               \n"
    "    int number1 = random(gl_InstanceID, 3);                                                                 \n"
    "    int number2 = random(number1, 2);                                                                       \n"
    "    offset += vec4(float(number1 & 0xFF) / 256.0,                                                           \n"
    "                   0.0f,                                                                                    \n"
    "                   float(number2 & 0xFF) / 256.0,                                                           \n"
    "                   0.0f);                                                                                   \n"
    "    float angle = float(random(number2, 2) & 0x3FF) / 1024.0;                                               \n"
    "                                                                                                            \n"
    "    vec2 texcoord = offset.xz / 1024.0 + vec2(0.5);                                                         \n"
    "                                                                                                            \n"
    "    float bend_factor = float(random(number2, 7) & 0x3FF) / 1024.0;                                         \n"
    "    float bend_amount = cos(vVertex.y);                                                                     \n"
    "                                                                                                            \n"
    "    mat4 rot = construct_rotation_matrix(angle);                                                            \n"
    "    vec4 position = (rot * (vVertex + vec4(0.0, 0.0, bend_amount * bend_factor, 0.0))) + offset;            \n"
    "                                                                                                            \n"
    "    position *= vec4(1.0, float(number1 & 0xFF) / 256.0 * 0.9 + 0.3, 1.0, 1.0);                             \n"
    "                                                                                                            \n"
    "    gl_Position = mvpMatrix * position;                                                                     \n"
    "    color = vec4(random_vector(gl_InstanceID).xyz * vec3(0.1, 0.5, 0.1) + vec3(0.1, 0.4, 0.1), 1.0);        \n"
    "}                                                                                                           \n"
};

static const char * grass_fs_source[] =
{
    "#version 410 core                \n"
    "                                 \n"
    "in vec4 color;                   \n"
    "                                 \n"
    "out vec4 output_color;           \n"
    "                                 \n"
    "void main(void)                  \n"
    "{                                \n"
    "    output_color = color;        \n"
    "}                                \n"
};

GLuint grass_buffer;
GLuint grass_vao;
GLuint grass_program;
mat4 proj_matrix(1.0f);

struct
{
	GLint mvpMatrix;
} uniforms;

void My_Init()
{
    static const GLfloat grass_blade[] =
    {
        -0.3f, 0.0f,
         0.3f, 0.0f,
        -0.20f, 1.0f,
         0.1f, 1.3f,
        -0.05f, 2.3f,
         0.0f, 3.3f
    };

    glGenBuffers(1, &grass_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, grass_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(grass_blade), grass_blade, GL_STATIC_DRAW);

    glGenVertexArrays(1, &grass_vao);
    glBindVertexArray(grass_vao);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    grass_program = glCreateProgram();
    GLuint grass_vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint grass_fs = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(grass_vs, 1, grass_vs_source, NULL);
    glShaderSource(grass_fs, 1, grass_fs_source, NULL);

    glCompileShader(grass_vs);
    glCompileShader(grass_fs);

    glAttachShader(grass_program, grass_vs);
    glAttachShader(grass_program, grass_fs);

    glLinkProgram(grass_program);
    glDeleteShader(grass_fs);
    glDeleteShader(grass_vs);

    uniforms.mvpMatrix = glGetUniformLocation(grass_program, "mvpMatrix");
}

void My_Display()
{
    float t = glutGet(GLUT_ELAPSED_TIME) * 0.00002f;
    float r = 550.0f;

    static const GLfloat black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    static const GLfloat one = 1.0f;
    glClearBufferfv(GL_COLOR, 0, black);
    glClearBufferfv(GL_DEPTH, 0, &one);

    mat4 mv_matrix = lookAt(vec3(sinf(t) * r, 25.0f, cosf(t) * r),
                                            vec3(0.0f, -50.0f, 0.0f),
                                            vec3(0.0, 1.0, 0.0));

    glUseProgram(grass_program);
    glUniformMatrix4fv(uniforms.mvpMatrix, 1, GL_FALSE, &(proj_matrix * mv_matrix)[0][0]);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glBindVertexArray(grass_vao);
	// void glDrawArraysInstanced(	GLenum mode,
	// 	GLint first,
	// 	GLsizei count,
	// 	GLsizei instancecount
	// );

    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 6, 1024 * 1024);

	glutSwapBuffers();
}

void My_Reshape(int width, int height)
{
	glViewport(0, 0, width, height);
	float viewportAspect = (float)width / (float)height;
	proj_matrix = perspective(deg2rad(45.0f), viewportAspect, 0.1f, 1000.0f);
}

void My_Timer(int val)
{
	glutPostRedisplay();
	glutTimerFunc(16, My_Timer, val);
}

int main(int argc, char *argv[])
{
    // Change working directory to source code path
    chdir(__FILEPATH__);
	// Initialize GLUT and GLEW, then create a window.
	////////////////////
	glutInit(&argc, argv);
#ifdef _MSC_VER
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#else
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(600, 600);
	glutCreateWindow(__FILENAME__); // You cannot use OpenGL functions before this line; The OpenGL context must be created first by glutCreateWindow()!
#ifdef _MSC_VER
	glewInit();
#endif
	printGLContextInfo();
	My_Init();
	////////////////////

	// Register GLUT callback functions.
	///////////////////////////////
	glutDisplayFunc(My_Display);
	glutReshapeFunc(My_Reshape);
	glutTimerFunc(16, My_Timer, 0);
	///////////////////////////////

	// Enter main event loop.
	//////////////
	glutMainLoop();
	//////////////
	return 0;
}
```

# Instanced_Attributes

```cpp
#include ".../.../Include/Common.h"

static const char * square_vs_source[] =
{
    "#version 410 core                                                               \n"
    "                                                                                \n"
    "layout (location = 0) in vec4 position;                                         \n"
    "layout (location = 1) in vec4 instance_color;                                   \n"
    "layout (location = 2) in vec4 instance_position;                                \n"
    "                                                                                \n"
    "out Fragment                                                                    \n"
    "{                                                                               \n"
    "    vec4 color;                                                                 \n"
    "} fragment;                                                                     \n"
    "                                                                                \n"
    "void main(void)                                                                 \n"
    "{                                                                               \n"
    "    gl_Position = (position + instance_position) * vec4(0.25, 0.25, 1.0, 1.0);  \n"
    "    fragment.color = instance_color;                                            \n"
    "}                                                                               \n"
};

static const char * square_fs_source[] =
{
    "#version 410 core                                                                \n"
    "precision highp float;                                                           \n"
    "                                                                                 \n"
    "in Fragment                                                                      \n"
    "{                                                                                \n"
    "    vec4 color;                                                                  \n"
    "} fragment;                                                                      \n"
    "                                                                                 \n"
    "out vec4 color;                                                                  \n"
    "                                                                                 \n"
    "void main(void)                                                                  \n"
    "{                                                                                \n"
    "    color = fragment.color;                                                      \n"
    "}                                                                                \n"
};

GLuint      square_buffer;
GLuint      square_vao;
GLuint      square_program;


void My_Init()
{
    static const GLfloat square_vertices[] =
    {
        -1.0f, -1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 0.0f, 1.0f,
            1.0f,  1.0f, 0.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };

    static const GLfloat instance_colors[] =
    {
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 0.0f, 1.0f
    };

    static const GLfloat instance_positions[] =
    {
        -2.0f, -2.0f, 0.0f, 0.0f,
            2.0f, -2.0f, 0.0f, 0.0f,
            2.0f,  2.0f, 0.0f, 0.0f,
        -2.0f,  2.0f, 0.0f, 0.0f
    };

    GLuint offset = 0;

    glGenVertexArrays(1, &square_vao);
    glGenBuffers(1, &square_buffer);
    glBindVertexArray(square_vao);
    glBindBuffer(GL_ARRAY_BUFFER, square_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(square_vertices) + sizeof(instance_colors) + sizeof(instance_positions), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(square_vertices), square_vertices);
    offset += sizeof(square_vertices);
    glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(instance_colors), instance_colors);
    offset += sizeof(instance_colors);
    glBufferSubData(GL_ARRAY_BUFFER, offset, sizeof(instance_positions), instance_positions);
    offset += sizeof(instance_positions);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)sizeof(square_vertices));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)(sizeof(square_vertices) + sizeof(instance_colors)));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glVertexAttribDivisor(1, 1); // position = 1 的那個頻率是 1，每個instance使用1次，2，如果是每兩個instance都用一樣的
    glVertexAttribDivisor(2, 1) ;// position = 2 的那個頻率是 1，每個instance使用1次

    square_program = glCreateProgram();

    GLuint square_vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(square_vs, 1, square_vs_source, NULL);
    glCompileShader(square_vs);
    glAttachShader(square_program, square_vs);
    GLuint square_fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(square_fs, 1, square_fs_source, NULL);
    glCompileShader(square_fs);
    glAttachShader(square_program, square_fs);

    glLinkProgram(square_program);
    glDeleteShader(square_vs);
    glDeleteShader(square_fs);
}

void My_Display()
{
    static const GLfloat black[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    glClearBufferfv(GL_COLOR, 0, black);

    glUseProgram(square_program);
    glBindVertexArray(square_vao);
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, 4);

	glutSwapBuffers();
}

void My_Reshape(int width, int height)
{
	glViewport(0, 0, width, height);
}

int main(int argc, char *argv[])
{
    // Change working directory to source code path
    chdir(__FILEPATH__);
	// Initialize GLUT and GLEW, then create a window.
	////////////////////
	glutInit(&argc, argv);
#ifdef _MSC_VER
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#else
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(600, 600);
	glutCreateWindow(__FILENAME__); // You cannot use OpenGL functions before this line; The OpenGL context must be created first by glutCreateWindow()!
#ifdef _MSC_VER
	glewInit();
#endif
	printGLContextInfo();
	My_Init();
	////////////////////

	// Register GLUT callback functions.
	///////////////////////////////
	glutDisplayFunc(My_Display);
	glutReshapeFunc(My_Reshape);
	///////////////////////////////

	// Enter main event loop.
	//////////////
	glutMainLoop();
	//////////////
	return 0;
}
```

# Point Sprite

```cpp
#include ".../.../Include/Common.h"
#define MENU_TIMER_START 1
#define MENU_TIMER_STOP 2
#define MENU_EXIT 3
#define M_PI = 3.14;

GLubyte timer_cnt = 0;
bool timer_enabled = true;
unsigned int timer_speed = 16;
using namespace std;
using namespace glm;

mat4 mvp(1.0f);
GLint um4mvp;
GLuint m_texture;
static unsigned int seed = 0x13371337;

static inline float random_float()
{
	float res;
	unsigned int tmp;

	seed *= 16807;

	tmp = seed ^ (seed >> 4) ^ (seed << 15);

	*((unsigned int *) &res) = (tmp >> 9) | 0x3F800000;

	return (res - 1.0f);
}

enum
{
	NUM_STARS           = 1000
};


static const char * vs_source[] =
{
	"#version 410 core                                              \n"
	"                                                               \n"
	"layout (location = 0) in vec4 position;                        \n"
	"layout (location = 1) in vec4 color;                           \n"
	"                                                               \n"
	"uniform float time;                                            \n"
	"uniform mat4 proj_matrix;                                      \n"
	"                                                               \n"
	"flat out vec4 starColor;                                       \n"
	"                                                               \n"
	"void main(void)                                                \n"
	"{                                                              \n"
	"    vec4 newVertex = position;                                 \n"
	"                                                               \n"
	"    newVertex.z += time;                                       \n"
	"    newVertex.z = fract(newVertex.z);                          \n"
	"                                                               \n"
	"    float size = (20.0 * newVertex.z * newVertex.z);           \n"
	"                                                               \n"
	"    starColor = smoothstep(1.0, 7.0, size) * color;            \n"
	"                                                               \n"
	"    newVertex.z = (999.9 * newVertex.z) - 1000.0;              \n"
	"    gl_Position = proj_matrix * newVertex;                     \n"
	"    gl_PointSize = size;                                       \n"
	"}                                                              \n"
};

static const char * fs_source[] =
{
	"#version 410 core                                              \n"
	"                                                               \n"
	"out vec4 color;										        \n"
	"                                                               \n"
	"uniform sampler2D tex_star;									\n"
	"flat in vec4 starColor;                                        \n"
	"                                                               \n"
	"void main(void)                                                \n"
	"{                                                              \n"
	"    color = texture(tex_star,gl_PointCoord);				    \n"
	"}                                                              \n"
};


GLuint          program;
GLuint          vao;
GLuint			vertex_shader;
GLuint			fragment_shader;
GLuint          buffer;
GLint           mv_location;
GLint           proj_location;
GLint			time_Loc;

mat4 proj_matrix(1.0f);

void My_Init()
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	program = glCreateProgram();
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, fs_source, NULL);
	glCompileShader(fs);
	printGLShaderLog(fs);
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, vs_source, NULL);
	glCompileShader(vs);

	glAttachShader(program, vs);
	glAttachShader(program, fs);

	glLinkProgram(program);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);

	proj_location = glGetUniformLocation(program, "proj_matrix");
	time_Loc = glGetUniformLocation(program, "time");

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	struct star_t
	{
		vec3 position;
		vec3 color;
	};

	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, NUM_STARS * sizeof(star_t), NULL, GL_STATIC_DRAW);

	// void *glMapBufferRange(
	//  GLenum target,
	// 	GLintptr offset,
	// 	GLsizeiptr length,
	// 	GLbitfield access
	//);

	star_t * star = (star_t *)glMapBufferRange(GL_ARRAY_BUFFER, 0, NUM_STARS * sizeof(star_t), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	int i;

	for (i = 0; i < 1000; i++)
	{
		star[i].position[0] = (random_float() * 2.0f - 1.0f) * 100.0f;
		star[i].position[1] = (random_float() * 2.0f - 1.0f) * 100.0f;
		star[i].position[2] = random_float();
		star[i].color[0] = 0.8f + random_float() * 0.2f;
		star[i].color[1] = 0.8f + random_float() * 0.2f;
		star[i].color[2] = 0.8f + random_float() * 0.2f;
	}



	glUnmapBuffer(GL_ARRAY_BUFFER);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(star_t), NULL);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(star_t), (void *)sizeof(vec3));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnable(GL_TEXTURE_2D);
	glActiveTexture( GL_TEXTURE0 );

	glEnable(GL_POINT_SPRITE); //點精靈會根據視點自動調整其大小和方向，這樣在三維空間中，它們看起來更自然。
	TextureData tdata = loadImg(".../.../Media/Textures/star.png");

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glGenTextures( 1, &m_texture );
	glBindTexture( GL_TEXTURE_2D, m_texture);
	//glTexImage2D(..., mipmap, texture format, ..., ..., 邊框, 來源格式, 通道, ...)
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, tdata.width, tdata.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tdata.data );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ); //水平
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );;

}

// GLUT callback. Called to draw the scene.
void My_Display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	static const GLfloat black[] = { 0.0f, 0, 0.0f, 1.0f };
	static const GLfloat one = 1.0f;

	glClearBufferfv(GL_COLOR, 0, black);
	glClearBufferfv(GL_DEPTH, 0, &one);

	glUseProgram(program);

	float f_timer_cnt = glutGet(GLUT_ELAPSED_TIME);
	float currentTime = f_timer_cnt* 0.001f;

	currentTime *= 0.1f;
	currentTime -= floor(currentTime);

	glUniform1f(time_Loc, currentTime);
	glUniformMatrix4fv(proj_location, 1, GL_FALSE, &proj_matrix[0][0]);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, m_texture);
	glEnable(GL_PROGRAM_POINT_SIZE);//，OpenGL 會允許頂點著色器使用 gl_PointSize 變量來動態控制每個點的大小。
	glDrawArrays(GL_POINTS, 0, NUM_STARS);

	glutSwapBuffers();
}

void My_Reshape(int width, int height)
{
	glViewport(0, 0, width, height);

	float viewportAspect = (float)width / (float)height;

	proj_matrix = perspective(deg2rad(50.0f), viewportAspect, 0.1f, 1000.0f);

}

void My_Timer(int val)
{
	timer_cnt++;
	glutPostRedisplay();
	if(timer_enabled)
	{
		glutTimerFunc(timer_speed, My_Timer, val);
	}
}

int main(int argc, char *argv[])
{
    // Change working directory to source code path
    chdir(__FILEPATH__);
	// Initialize GLUT and GLEW, then create a window.
	////////////////////
	glutInit(&argc, argv);
#ifdef _MSC_VER
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#else
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(600, 600);
	glutCreateWindow(__FILENAME__); // You cannot use OpenGL functions before this line; The OpenGL context must be created first by glutCreateWindow()!
#ifdef _MSC_VER
	glewInit();
#endif
	printGLContextInfo();
	My_Init();
	////////////////////

	// Register GLUT callback functions.
	///////////////////////////////
	glutDisplayFunc(My_Display);
	glutReshapeFunc(My_Reshape);
	glutTimerFunc(timer_speed, My_Timer, 0);
	///////////////////////////////

	// Enter main event loop.
	//////////////
	glutMainLoop();
	//////////////
	return 0;
}
```

# GrayScale_Cube

```cpp
#include ".../.../Include/Common.h"
#define GLM_SWIZZLE

#include <cstdio>
#include <cstdlib>

#define MENU_TIMER_START 1
#define MENU_TIMER_STOP 2
#define MENU_EXIT 3
#define M_PI = 3.14;

GLubyte timer_cnt = 0;
bool timer_enabled = true;
unsigned int timer_speed = 16;
using namespace std;
using namespace glm;

mat4 mvp(1.0f);
GLint um4mvp;


static const char * vs_source[] =
{
	"#version 410 core                                                  \n"
	"                                                                   \n"
	"in vec4 position;                                                  \n"
	"                                                                   \n"
	"out VS_OUT                                                         \n"
	"{                                                                  \n"
	"    vec4 color;                                                    \n"
	"} vs_out;                                                          \n"
	"                                                                   \n"
	"uniform mat4 mv_matrix;                                            \n"
	"uniform mat4 proj_matrix;                                          \n"
	"                                                                   \n"
	"void main(void)                                                    \n"
	"{                                                                  \n"
	"    gl_Position = proj_matrix * mv_matrix * position;              \n"
	"    vs_out.color = position * 2.0 + vec4(0.5, 0.5, 0.5, 0.0);      \n"
	"}                                                                  \n"
};

static const char * fs_source[] =
{
	"#version 410 core                                                  \n"
	"                                                                   \n"
	"out vec4 color;                                                    \n"
	"                                                                   \n"
	"in VS_OUT                                                          \n"
	"{                                                                  \n"
	"    vec4 color;                                                   \n"
	"} fs_in;                                                           \n"
	"                                                                   \n"
	"void main(void)                                                    \n"
	"{                                                                  \n"
	"    color = fs_in.color;                                           \n"
	"}                                                                  \n"
};


static const char * vs_source2[] =
{
	"#version 410 core                                                  \n"
	"                                                                   \n"
	"layout (location = 0) in vec2 position;                            \n"
	"layout (location = 1) in vec2 texcoord;                            \n"
	"out VS_OUT                                                         \n"
	"{                                                                  \n"
	"    vec2 texcoord;                                                 \n"
	"} vs_out;                                                          \n"
	"                                                                   \n"
	"                                                                   \n"
	"void main(void)                                                    \n"
	"{                                                                  \n"
	"    gl_Position = vec4(position,0.0,1.0);							\n"
	"    vs_out.texcoord = texcoord;                                    \n"
	"}																	\n"
};


static const char * fs_source2[] =
{
	"#version 410 core                                                              \n"
	"                                                                               \n"
	"uniform sampler2D tex;                                                         \n"
	"                                                                               \n"
	"out vec4 color;                                                                \n"
	"                                                                               \n"
	"in VS_OUT                                                                      \n"
	"{                                                                              \n"
	"    vec2 texcoord;                                                             \n"
	"} fs_in;                                                                       \n"
	"                                                                               \n"
	"void main(void)                                                                \n"
	"{                                                                              \n"
	"    vec4 texture_color = texture(tex,fs_in.texcoord);							\n"
	"	 float grayscale_color = 0.2126*texture_color.r+0.7152*texture_color.g+0.0722*texture_color.b; \n"
	"    color = vec4(grayscale_color,grayscale_color,grayscale_color,1.0);			\n"
	"}                                                                              \n"
};
static const GLfloat vertex_positions[] =
{
	-0.25f,  0.25f, -0.25f,
	-0.25f, -0.25f, -0.25f,
	0.25f, -0.25f, -0.25f,

	0.25f, -0.25f, -0.25f,
	0.25f,  0.25f, -0.25f,
	-0.25f,  0.25f, -0.25f,

	0.25f, -0.25f, -0.25f,
	0.25f, -0.25f,  0.25f,
	0.25f,  0.25f, -0.25f,

	0.25f, -0.25f,  0.25f,
	0.25f,  0.25f,  0.25f,
	0.25f,  0.25f, -0.25f,

	0.25f, -0.25f,  0.25f,
	-0.25f, -0.25f,  0.25f,
	0.25f,  0.25f,  0.25f,

	-0.25f, -0.25f,  0.25f,
	-0.25f,  0.25f,  0.25f,
	0.25f,  0.25f,  0.25f,

	-0.25f, -0.25f,  0.25f,
	-0.25f, -0.25f, -0.25f,
	-0.25f,  0.25f,  0.25f,

	-0.25f, -0.25f, -0.25f,
	-0.25f,  0.25f, -0.25f,
	-0.25f,  0.25f,  0.25f,

	-0.25f, -0.25f,  0.25f,
	0.25f, -0.25f,  0.25f,
	0.25f, -0.25f, -0.25f,

	0.25f, -0.25f, -0.25f,
	-0.25f, -0.25f, -0.25f,
	-0.25f, -0.25f,  0.25f,

	-0.25f,  0.25f, -0.25f,
	0.25f,  0.25f, -0.25f,
	0.25f,  0.25f,  0.25f,

	0.25f,  0.25f,  0.25f,
	-0.25f,  0.25f,  0.25f,
	-0.25f,  0.25f, -0.25f
};

static const GLfloat window_positions[] =
{
	1.0f,-1.0f,1.0f,0.0f,
	-1.0f,-1.0f,0.0f,0.0f,
	-1.0f,1.0f,0.0f,1.0f,
	1.0f,1.0f,1.0f,1.0f
};

GLuint          program;
GLuint			program2;
GLuint          vao;
GLuint          window_vao;
GLuint			vertex_shader;
GLuint			fragment_shader;
GLuint          buffer;
GLuint			window_buffer;
GLint           mv_location;
GLint           proj_location;

GLuint			FBO;
GLuint			depthRBO;
GLuint	FBODataTexture;

mat4 proj_matrix(1.0f);

void My_Init()
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	program = glCreateProgram();
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, fs_source, NULL);
	glCompileShader(fs);

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, vs_source, NULL);
	glCompileShader(vs);

	glAttachShader(program, vs);
	glAttachShader(program, fs);

	glLinkProgram(program);

	program2 = glCreateProgram();

	GLuint vs2 = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs2, 1, vs_source2, NULL);
	glCompileShader(vs2);
	printGLShaderLog(vs2);

	GLuint fs2 = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs2, 1, fs_source2, NULL);
	glCompileShader(fs2);
	printGLShaderLog(fs2);

	glAttachShader(program2, vs2);
	glAttachShader(program2, fs2);

	glLinkProgram(program2);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_positions), vertex_positions, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	mv_location = glGetUniformLocation(program, "mv_matrix");
	proj_location = glGetUniformLocation(program, "proj_matrix");

	glGenVertexArrays(1, &window_vao);
	glBindVertexArray(window_vao);

	glGenBuffers(1, &window_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, window_buffer);
	glBufferData(GL_ARRAY_BUFFER,sizeof(window_positions),window_positions,	GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT)*4, 0);
	glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT)*4, (const GLvoid*)(sizeof(GL_FLOAT)*2));

	glEnableVertexAttribArray( 0 );
	glEnableVertexAttribArray( 1 );

	glGenFramebuffers( 1, &FBO );

	//////////////////////////////////////////////////////////////////////////
	/////////Create RBO and Render Texture in Reshape Function////////////////
	//////////////////////////////////////////////////////////////////////////
}

// GLUT callback. Called to draw the scene.
void My_Display()
{
	glBindTexture( GL_TEXTURE_2D, 0 );
	glBindFramebuffer( GL_DRAW_FRAMEBUFFER, FBO );
	glDrawBuffer( GL_COLOR_ATTACHMENT0 );

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	static const GLfloat green[] = { 0.0f, 0.25f, 0.0f, 1.0f };
	static const GLfloat one = 1.0f;

	glClearBufferfv(GL_COLOR, 0, green);
	glClearBufferfv(GL_DEPTH, 0, &one);

	glUseProgram(program);

	glBindVertexArray(vao);
	glUniformMatrix4fv(proj_location, 1, GL_FALSE, &proj_matrix[0][0]);

	mat4 Identy_Init(1.0f);

	float f_timer_cnt = glutGet(GLUT_ELAPSED_TIME);
	float currentTime = f_timer_cnt* 0.001f;
	float f = (float)currentTime * 0.3f;

	mat4 mv_matrix = translate(Identy_Init, vec3(0.0f, 0.0f, -4.0f));

	mv_matrix = translate(mv_matrix, vec3(sinf(2.1f * f) * 0.5f,cosf(1.7f * f) * 0.5f,	sinf(1.3f * f) * cosf(1.5f * f) * 2.0f));

	mv_matrix = rotate(mv_matrix,deg2rad(currentTime*45.0f), vec3(0.0f, 1.0f, 0.0f));
	mv_matrix = rotate(mv_matrix,deg2rad(currentTime*81.0f), vec3(1.0f, 0.0f, 0.0f));

	glUniformMatrix4fv(mv_location, 1, GL_FALSE, &mv_matrix[0][0]);
	glDrawArrays(GL_TRIANGLES, 0, 36);

	glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0 );

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glClearColor( 1.0f, 0.0f, 0.0f, 1.0f );

	glActiveTexture(GL_TEXTURE0);
	glBindTexture( GL_TEXTURE_2D, FBODataTexture );
	glBindVertexArray(window_vao);
	glUseProgram(program2);
	glDrawArrays(GL_TRIANGLE_FAN,0,4 );
	glutSwapBuffers();
}
// FRAMEBUFFER
void My_Reshape(int width, int height)
{
	glViewport(0, 0, width, height);

	float viewportAspect = (float)width / (float)height;
	proj_matrix = perspective(deg2rad(60.0f), viewportAspect, 0.1f, 1000.0f);

	glDeleteRenderbuffers(1,&depthRBO);
	glDeleteTextures(1,&FBODataTexture);
	glGenRenderbuffers( 1, &depthRBO );
	glBindRenderbuffer( GL_RENDERBUFFER, depthRBO );
	glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, width, height );

	glGenTextures( 1, &FBODataTexture );
	glBindTexture( GL_TEXTURE_2D, FBODataTexture);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

	glBindFramebuffer( GL_DRAW_FRAMEBUFFER, FBO );
	glFramebufferRenderbuffer( GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO );
	// void glFramebufferTexture2D(	GLenum target,
 	// GLenum attachment,
 	// GLenum textarget,
 	// GLuint texture,
 	// GLint level);
	glFramebufferTexture2D( GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FBODataTexture, 0 );

}

void My_Timer(int val)
{
	timer_cnt++;
	glutPostRedisplay();
	if(timer_enabled)
	{
		glutTimerFunc(timer_speed, My_Timer, val);
	}
}

int main(int argc, char *argv[])
{
    // Change working directory to source code path
    chdir(__FILEPATH__);
	// Initialize GLUT and GLEW, then create a window.
	////////////////////
	glutInit(&argc, argv);
#ifdef _MSC_VER
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#else
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(600, 600);
	glutCreateWindow(__FILENAME__); // You cannot use OpenGL functions before this line; The OpenGL context must be created first by glutCreateWindow()!
#ifdef _MSC_VER
	glewInit();
#endif
	printGLContextInfo();
	My_Init();
	////////////////////

	// Register GLUT callback functions.
	///////////////////////////////
	glutDisplayFunc(My_Display);
	glutReshapeFunc(My_Reshape);
	glutTimerFunc(timer_speed, My_Timer, 0);
	///////////////////////////////

	// Enter main event loop.
	//////////////
	glutMainLoop();
	//////////////
	return 0;
}
```

# Image Processing

```cpp
#include ".../.../Include/Common.h"

#define MENU_TIMER_START 1
#define MENU_TIMER_STOP 2
#define MENU_EXIT 3

#define Shader_Blur 4
#define Shader_Quantization 5
#define Shader_DoG 6

int shader_now = 0;

GLubyte timer_cnt = 0;
bool timer_enabled = true;
unsigned int timer_speed = 16;

using namespace glm;

mat4 mvp(1.0f);
GLint um4mvp;

GLuint hawk_texture;
GLint Shader_now_Loc;
int defalut_w = 800;
int defalut_h = 600;

static const char * vs_source[] =
{
	"#version 410 core                                                  \n"
	"                                                                   \n"
	"layout (location = 0) in vec2 position;                            \n"
	"layout (location = 1) in vec2 texcoord;                            \n"
	"out VS_OUT                                                         \n"
	"{                                                                  \n"
	"    vec2 texcoord;                                                 \n"
	"} vs_out;                                                          \n"
	"                                                                   \n"
	"                                                                   \n"
	"void main(void)                                                    \n"
	"{                                                                  \n"
	"    gl_Position = vec4(position,0.0,1.0);							\n"
	"    vs_out.texcoord = texcoord;                                    \n"
	"}																	\n"
};

static const char * fs_source[] =
{
	"#version 410\n"
	"uniform sampler2D tex; \n"
	"out vec4 color;\n"
	"uniform int shader_now ;\n"
	"in VS_OUT\n"
	"{\n"
	"   vec2 texcoord;\n"
	"} fs_in;\n"
	"float sigma_e = 2.0f;\n"
	"float sigma_r = 2.8f;\n"
	"float phi = 3.4f;\n"
	"float tau = 0.99f;\n"
	"float twoSigmaESquared = 2.0 * sigma_e * sigma_e;		\n"
	"float twoSigmaRSquared = 2.0 * sigma_r * sigma_r;		\n"
	"int halfWidth = int(ceil( 2.0 * sigma_r ));\n"
	"vec2 img_size = vec2(1024,768);\n"
	"int nbins = 8;\n"
	"void main(void)\n"
	"{\n"
	" \n"
	"	switch(shader_now)\n"
	"	{\n"
	"		case(2):\n"
	"			{\n"
	"				\n"
	"				vec2 sum = vec2(0.0);\n"
	"				vec2 norm = vec2(0.0);\n"
	"				int kernel_count = 0;\n"
	"			for ( int i = -halfWidth; i <= halfWidth; ++i ) {\n"
	"			for ( int j = -halfWidth; j <= halfWidth; ++j ) {\n"
	"					float d = length(vec2(i,j));\n"
	"					vec2 kernel = vec2( exp( -d * d / twoSigmaESquared ), \n"
	"										exp( -d * d / twoSigmaRSquared ));\n"
	"					vec4 c = texture(tex, fs_in.texcoord + vec2(i,j) / img_size);\n"
	"					vec2 L = vec2(0.299 * c.r + 0.587 * c.g + 0.114 * c.b);\n"
	"														\n"
	"					norm += 2.0 * kernel;\n"
	"					sum += kernel * L;\n"
	"				}\n"
	"			}\n"
	"			sum /= norm;\n"
	"			\n"
	"			float H = 100.0 * (sum.x - tau * sum.y);\n"
	"			float edge = ( H > 0.0 )? 1.0 : 2.0 * smoothstep(-2.0, 2.0, phi * H );\n"
	"				\n"
	"		   color = vec4(edge,edge,edge,1.0 );\n"
	"				break;\n"
	"			}\n"
	"		case(1):\n"
	"			{\n"
	"				vec4 texture_color = texture(tex,fs_in.texcoord);\n"
	"   \n"
	"			float r = floor(texture_color.r * float(nbins)) / float(nbins);\n"
	"			 float g = floor(texture_color.g * float(nbins)) / float(nbins);\n"
	"			float b = floor(texture_color.b * float(nbins)) / float(nbins); \n"
	"			color = vec4(r,g,b,texture_color.a);\n"
	"				break;\n"
	"			}\n"
	"		case(0):\n"
	"			{\n"
	"				color = vec4(0);	\n"
	"			int n = 0;\n"
	"			int half_size = 3;\n"
	"			for ( int i = -half_size; i <= half_size; ++i ) {        \n"
	"				for ( int j = -half_size; j <= half_size; ++j ) {\n"
	"					 vec4 c = texture(tex, fs_in.texcoord + vec2(i,j)/img_size); \n"
	"					 color+= c;\n"
	"				     n++;\n"
	"					}\n"
	"				}\n"
	"				color /=n;\n"
	"					break;\n"
	"			}\n"
	"			\n"
	"	\n"
	"	}\n"
	"}\n"

};


void My_Init()
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	GLuint program = glCreateProgram();
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vertexShader, 1, vs_source, NULL);
	glShaderSource(fragmentShader, 1, fs_source, NULL);

	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);
	printGLShaderLog(vertexShader);
	printGLShaderLog(fragmentShader);
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);
	um4mvp = glGetUniformLocation(program, "um4mvp");
	glUseProgram(program);

	Shader_now_Loc = glGetUniformLocation(program, "shader_now");

	GLuint buffer;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, 0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void*)(sizeof(float) * 2));

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	float data[] = {
		1.0f,-1.0f,1.0f,0.0f,
		-1.0f,-1.0f,0.0f,0.0f,
		-1.0f,1.0f,0.0f,1.0f,
		1.0f,1.0f,1.0f,1.0f
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

	TextureData tdata = loadImg(".../.../Media/Textures/hawk.png");

	glGenTextures( 1, &hawk_texture );
	glBindTexture( GL_TEXTURE_2D, hawk_texture);
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, tdata.width, tdata.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tdata.data );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	///////////////////////////////////////////////////////////////////////////
	printf("\nNote : Use Right Click Menu to switch Effect\n");
	//////////////////////////////////////////////////////////////////////////
}

// GLUT callback. Called to draw the scene.
void My_Display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	float f_timer_cnt = timer_cnt / 255.0f;


	glUniform1i(Shader_now_Loc,shader_now);
	glUniformMatrix4fv(um4mvp, 1, GL_FALSE, value_ptr(mvp));
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glutSwapBuffers();
}

void My_Reshape(int width, int height)
{
	glViewport(0, 0, width, height);

	float viewportAspect = (float)width / (float)height;
	mvp = ortho(-1 * viewportAspect, 1 * viewportAspect, -1.0f, 1.0f);
	mvp = mvp * lookAt(vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
}

void My_Timer(int val)
{
	timer_cnt++;
	glutPostRedisplay();
	if(timer_enabled)
	{
		glutTimerFunc(timer_speed, My_Timer, val);
	}
}

void My_Menu(int id)
{
	switch(id)
	{
	case MENU_TIMER_START:
		if(!timer_enabled)
		{
			timer_enabled = true;
			glutTimerFunc(timer_speed, My_Timer, 0);
		}
		break;
	case MENU_TIMER_STOP:
		timer_enabled = false;
		break;
	case MENU_EXIT:
		exit(0);
		break;
	case Shader_Blur:
		shader_now = 0;
		break;
	case Shader_Quantization:
		shader_now = 1;
		break;
	case Shader_DoG:
		shader_now = 2;
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

int main(int argc, char *argv[])
{
    // Change working directory to source code path
    chdir(__FILEPATH__);
	// Initialize GLUT and GLEW, then create a window.
	////////////////////
	glutInit(&argc, argv);
#ifdef _MSC_VER
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#else
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(defalut_w, defalut_h);
	glutCreateWindow(__FILENAME__); // You cannot use OpenGL functions before this line; The OpenGL context must be created first by glutCreateWindow()!
#ifdef _MSC_VER
	glewInit();
#endif
	printGLContextInfo();
	My_Init();
	////////////////////

	// Create a menu and bind it to mouse right button.
	////////////////////////////
	int menu_main = glutCreateMenu(My_Menu);
	int menu_timer = glutCreateMenu(My_Menu);
	int shader = glutCreateMenu(My_Menu);

	glutSetMenu(menu_main);
	glutAddSubMenu("Timer", menu_timer);
	glutAddSubMenu("Shader", shader);

	glutAddMenuEntry("Exit", MENU_EXIT);

	glutSetMenu(menu_timer);
	glutAddMenuEntry("Start", MENU_TIMER_START);
	glutAddMenuEntry("Stop", MENU_TIMER_STOP);

	glutSetMenu(shader);
	glutAddMenuEntry("Blur", Shader_Blur);
	glutAddMenuEntry("Quantization", Shader_Quantization);
	glutAddMenuEntry("DoG", Shader_DoG);

	glutSetMenu(menu_main);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
	////////////////////////////

	// Register GLUT callback functions.
	///////////////////////////////
	glutDisplayFunc(My_Display);
	glutReshapeFunc(My_Reshape);
	glutTimerFunc(timer_speed, My_Timer, 0);
	///////////////////////////////

	// Enter main event loop.
	//////////////
	glutMainLoop();
	//////////////
	return 0;
}
```
