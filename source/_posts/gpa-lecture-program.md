---
title: Graphcis Programming and Application Lecture Program 解析
date: 2024-10-06 20:43:02
tags: 
category: 
math: true
---

# Alienrain

## 前置作業和 Shaders
```cpp
#include "../../Include/Common.h"

using namespace glm;

static inline float random_float()
{
	float res;
	unsigned int tmp;

	static unsigned int seed = 0x13371337;
	seed *= 16807;

	tmp = seed ^ (seed >> 4) ^ (seed << 15);

	*((unsigned int *) &res) = (tmp >> 9) | 0x3F800000;

	return (res - 1.0f);
}


static const char * vs_source[] =
{
	"#version 410 core                                                      \n"
	"                                                                       \n"
	"layout (location = 1) in int alien_index;                              \n"
	"                                                                       \n"
	"out VS_OUT                                                             \n"
	"{                                                                      \n"
	"    flat int alien;                                                    \n"
	"    vec2 tc;                                                           \n"
	"} vs_out;                                                              \n"
	"                                                                       \n"
	"layout(std140) uniform droplets                                        \n"
	"{                                                                      \n"
	"    vec4 droplet[256];                                                 \n"
	"};                                                                     \n"
	"                                                                       \n"
	"void main(void)                                                        \n"
	"{                                                                      \n"
	"    const vec2[4] position = vec2[4](vec2(-0.5, -0.5),                 \n"
	"                                     vec2( 0.5, -0.5),                 \n"
	"                                     vec2(-0.5,  0.5),                 \n"
	"                                     vec2( 0.5,  0.5));                \n"
    "    const vec2[4] texcoord = vec2[4](vec2(0, 0),                       \n"
    "                                     vec2(1, 0),                       \n"
    "                                     vec2(0, 1),                       \n"
    "                                     vec2(1, 1));                      \n"
    "    vs_out.tc = texcoord[gl_VertexID];                                 \n"
	"    float co = cos(droplet[alien_index].z);                            \n"
	"    float so = sin(droplet[alien_index].z);                            \n"
	"    mat2 rot = mat2(vec2(co, so),                                      \n"
	"                    vec2(-so, co));                                    \n"
	"    vec2 pos = 0.25 * rot * position[gl_VertexID];                     \n"
	"    gl_Position = vec4(pos + droplet[alien_index].xy, 0.5, 1.0);       \n"
	"    vs_out.alien = int(mod(float(alien_index), 64.0));          		\n"
	"}                                                                      \n"
};

static const char * fs_source[] =
{
	"#version 410 core                                                      \n"
	"                                                                       \n"
	"layout (location = 0) out vec4 color;                                  \n" // 輸出顏色的名字是可以隨便取的，重點是要在 location = 0 (預設也是)
	"                                                                       \n"
	"in VS_OUT                                                              \n"
	"{                                                                      \n"
	"    flat int alien;                                                    \n"
	"    vec2 tc;                                                           \n"
	"} fs_in;                                                               \n"
	"                                                                       \n"
	"uniform sampler2DArray tex_aliens;                                     \n"
	"                                                                       \n"
	"void main(void)                                                        \n"
	"{                                                                      \n"
	"    color = texture(tex_aliens, vec3(fs_in.tc, float(fs_in.alien)));   \n"
	"}                                                                      \n"
};

GLuint          program;
GLuint          vao;
GLuint          tex_alien_array;
GLuint          rain_buffer;

float           droplet_x_offset[256];
float           droplet_rot_speed[256];
float           droplet_fall_speed[256];

```


```cpp
int main(int argc, char *argv[])
{
    chdir(__FILEPATH__);

	glutInit(&argc, argv); // GLUT INIT，最一開始一定要

#ifdef _MSC_VER // 微軟的預設編譯器 (MS: Microsoft，C: C 語言， VER: Version)
    // 要使用 RGBA color mode、Dobule-buffered 的 Window，還要啟動 Depth Buffer 來進行深度測試
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#else // 如果是用其他編譯器
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH); //先不管 應該差不多
#endif
	glutInitWindowPosition(100, 100); // 視窗出現的位置
	glutInitWindowSize(600, 600); // 視窗的長寬
	glutCreateWindow(__FILENAME__); // 視窗的名稱（一定要這行！不然不能跑）

#ifdef _MSC_VER 
	glewInit(); // glew init
#endif

	printGLContextInfo(); // custom function
	My_Init(); // custom function

	glutDisplayFunc(My_Display); // GLUT 的 callback function，也就是視窗顯示的東西
	glutReshapeFunc(My_Reshape); // 視窗 resize 會 call 的 function

    // glutTimerFunc(msecs,(*func)(int value), int value);，
	glutTimerFunc(16, My_Timer, 0); // 計時器，16 毫秒後會 call MY_Timer

	glutMainLoop(); // 進入 GLUT loop
	return 0;
}
```

```cpp

void My_Init()
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 把視窗變成白色
	glEnable(GL_DEPTH_TEST); // 啟動深度測試
	glDepthFunc(GL_LEQUAL); // GL_LEQUAL: 如果新的片段的深度直 <= 當前的片段深度值，那就可以蓋過去

	program = glCreateProgram(); // 新增一個 shader program，回傳 GLuint ID
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER); // 宣告他是 Fragment Shader
	glShaderSource(fs, 1, fs_source, NULL); // 寫入 shader 的內容
	glCompileShader(fs); // 把 shader 編譯成 GPU 可執行的程式 

	GLuint vs = glCreateShader(GL_VERTEX_SHADER); 
	glShaderSource(vs, 1, vs_source, NULL);
	glCompileShader(vs);

	glAttachShader(program, vs); // 綁定到 shader program
	glAttachShader(program, fs);
    printGLShaderLog(vs); // custom function，寫在 common.h 裡面
    printGLShaderLog(fs);

	glLinkProgram(program); // link shader program
	glUseProgram(program); // 啟動這個 program

	glGenVertexArrays(1, &vao); // glGenVertexArrays( vao 數量, vao GLuint ID)，生成 vao
	glBindVertexArray(vao); // 綁定到 vao，這樣之後才能寫入 attribute，像是 glVertexAttribI1i(1, alien_index);

	TextureData tex = loadImg("../../Media/Textures/aliens.png"); // custom function，回傳 TextureData，包含高、寬、字元
	glGenTextures(1, &tex_alien_array); // 生成 texture，寫 ID 到 text_alien_array
	glBindTexture(GL_TEXTURE_2D_ARRAY, tex_alien_array); // 綁定這個 ID，這邊用 GL_TEXTURE_2D_ARRAY，之後寫入宣告 GL_TEXTURE_2D_ARRAY 都會到對應的 ID

    // glTexImage3D(target, mipnap level, internalformat, width, height, depth, border, format, type, * data);
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, 256, 256, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.data); // 這邊 tex.data 是 256 * 16384，一次全 load 到 3D image 節省效能
	glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER,GL_LINEAR); // 設定縮小時要用 線性內插 的方式生成
    glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER,GL_LINEAR); // 放大時也用 線性內插
	delete[] tex.data;

	glGenBuffers(1, &rain_buffer); // 生成 buffer，綁到 rain_buffer 這個 ID
	glBindBuffer(GL_UNIFORM_BUFFER, rain_buffer); // 綁定 ID 是 GL_UNIFORM_BUFFER

    // glBufferData(target, size, data, usage)
	glBufferData(GL_UNIFORM_BUFFER, 256 * sizeof(vec4), NULL, GL_DYNAMIC_DRAW); // 初始化這個 rain_buffer，

    // glBindBufferBase(target,  index,  buffer);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, rain_buffer); // 連連看，buffer 連到 binding point index 0

    // 初始化位置、旋轉、速度
	for (int i = 0; i < 256; i++)
	{
		droplet_x_offset[i] = random_float() * 2.0f - 1.0f;
		droplet_rot_speed[i] = (random_float() + 0.5f) * ((i & 1) ? -3.0f : 3.0f);
		droplet_fall_speed[i] = random_float() + 0.2f;
	}

	glEnable(GL_BLEND); // 啟用混合功能，可以加入透明度
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); //如何混合的，源顏色的值 * alpha，目標顏色的值 * (1- 源顏色 alpha)
}
```

```cpp
// GLUT callback. Called to draw the scene.
void My_Display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 清除顏色、深度
	
	float currentTime = glutGet(GLUT_ELAPSED_TIME) * 0.001f; // glutGet(GLUT_ELAPSED_TIME) 可以得到程式從開始到現在的時間 (毫秒

    // glMapBufferRange(target, offset, length, access)
    // 這裡的 GL_UNIFORM_BUFFER 就是對應到前面寫的 rain_buffer
    // GL_MAP_WRITE_BIT: 允許寫入 map 到的 memory 
    // GL_MAP_INVALIDATE_BUFFER_BIT: map 前的內容可以被捨棄
	vec4 * droplet = (vec4 *)glMapBufferRange(GL_UNIFORM_BUFFER, 0, 256 * sizeof(vec4), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT); 
	for (int i = 0; i < 256; i++)
	{
		droplet[i][0] = droplet_x_offset[i];
        droplet[i][1] = 2.0f - fmodf((currentTime + float(i)) * droplet_fall_speed[i], 4.31f);
        droplet[i][2] = currentTime * droplet_rot_speed[i];
		droplet[i][3] = 0.0f; // padding
	}
	glUnmapBuffer(GL_UNIFORM_BUFFER); // 提交到 GPU

	int alien_index;
	for (alien_index = 0; alien_index < 256; alien_index++)
	{
		glVertexAttribI1i(1, alien_index); // location = 1, 
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // 從 0 開始畫四個 vertex
	}

	glutSwapBuffers();
}
```
```cpp
void My_Reshape(int width, int height)
{
	glViewport(0, 0, width, height);
}

void My_Timer(int val)
{
	glutPostRedisplay(); // 重畫
	glutTimerFunc(16, My_Timer, val);
}

```

# Simple_Texture_Coords
```cpp
#include "../../Include/Common.h"

using namespace glm;
using namespace std;

GLuint          program;
GLuint          tex_object[2];
GLuint          tex_index;
int index_count;
int vertex_count;
struct
{
    GLint       mv_matrix;
    GLint       proj_matrix;
} uniforms;

static const char *render_fs_glsl[] = 
{
    "#version 410 core                                            \n"
    "                                                             \n"
    "uniform sampler2D tex_object;                                \n"
    "                                                             \n"
    "in VS_OUT                                                    \n"
    "{                                                            \n"
    "    vec2 tc;                                                 \n"
    "} fs_in;                                                     \n"
    "                                                             \n"
    "out vec4 color;                                              \n"
    "                                                             \n"
    "void main(void)                                              \n"
    "{                                                            \n"
    "    color = texture(tex_object, fs_in.tc * vec2(3.0, 1.0));  \n"
    "}                                                            \n"
};

static const char *render_vs_glsl[] = 
{
    "#version 410 core                            \n"
    "                                             \n"
    "uniform mat4 mv_matrix;                      \n"
    "uniform mat4 proj_matrix;                    \n"
    "                                             \n"
    "layout (location = 0) in vec3 position;      \n"
    "layout (location = 1) in vec2 tc;            \n"
    "                                             \n"
    "out VS_OUT                                   \n"
    "{                                            \n"
    "    vec2 tc;                                 \n"
    "} vs_out;                                    \n"
    "                                             \n"
    "void main(void)                              \n"
    "{                                            \n"
    "    vec4 pos_vs = mv_matrix * vec4(position, 1.0);      \n"
    "                                             \n"
    "    vs_out.tc = tc;                          \n"
    "                                             \n"
    "    gl_Position = proj_matrix * pos_vs;      \n"
    "}                                            \n"
};

void My_Init()
{
	program = glCreateProgram();
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, render_fs_glsl, NULL);
	glCompileShader(fs);

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, render_vs_glsl, NULL);
	glCompileShader(vs);

	glAttachShader(program, vs);
	glAttachShader(program, fs);
	printGLShaderLog(vs);
	printGLShaderLog(fs);

	glLinkProgram(program);
	glUseProgram(program);

    uniforms.mv_matrix = glGetUniformLocation(program, "mv_matrix");
    uniforms.proj_matrix = glGetUniformLocation(program, "proj_matrix");

#define B 0x00, 0x00, 0x00, 0x00
#define W 0xFF, 0xFF, 0xFF, 0xFF
    static const GLubyte tex_data[] =
    {
        B, W, B, W, B, W, B, W, B, W, B, W, B, W, B, W,
        W, B, W, B, W, B, W, B, W, B, W, B, W, B, W, B,
        B, W, B, W, B, W, B, W, B, W, B, W, B, W, B, W,
        W, B, W, B, W, B, W, B, W, B, W, B, W, B, W, B,
        B, W, B, W, B, W, B, W, B, W, B, W, B, W, B, W,
        W, B, W, B, W, B, W, B, W, B, W, B, W, B, W, B,
        B, W, B, W, B, W, B, W, B, W, B, W, B, W, B, W,
        W, B, W, B, W, B, W, B, W, B, W, B, W, B, W, B,
        B, W, B, W, B, W, B, W, B, W, B, W, B, W, B, W,
        W, B, W, B, W, B, W, B, W, B, W, B, W, B, W, B,
        B, W, B, W, B, W, B, W, B, W, B, W, B, W, B, W,
        W, B, W, B, W, B, W, B, W, B, W, B, W, B, W, B,
        B, W, B, W, B, W, B, W, B, W, B, W, B, W, B, W,
        W, B, W, B, W, B, W, B, W, B, W, B, W, B, W, B,
        B, W, B, W, B, W, B, W, B, W, B, W, B, W, B, W,
        W, B, W, B, W, B, W, B, W, B, W, B, W, B, W, B,
    };
#undef B
#undef W

    glGenTextures(1, &tex_object[0]);
    glBindTexture(GL_TEXTURE_2D, tex_object[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, tex_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    TextureData tex = loadImg("../../Media/Textures/pattern1.png");
	glGenTextures(1, &tex_object[1]);
	glBindTexture(GL_TEXTURE_2D, tex_object[1]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width, tex.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	vector<MeshData> meshes;
	meshes = loadObj("../../Media/Objects/torus_nrms_tc.obj");
	
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	GLuint position_buffer;
	GLuint texcoord_buffer;
	GLuint index_buffer;

	glGenBuffers(1, &position_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, position_buffer);
	glBufferData(GL_ARRAY_BUFFER, meshes[0].positions.size() * sizeof(float), meshes[0].positions.data(), GL_STATIC_DRAW);


	// glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void * pointer);
	// index: attribute 位置 (location = 0)
	// size: vec3
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
	glEnableVertexAttribArray(0); // 啟動

	vertex_count = meshes[0].positions.size() / 3;

	glGenBuffers(1, &texcoord_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer);
	glBufferData(GL_ARRAY_BUFFER, meshes[0].texcoords.size() * sizeof(float), meshes[0].texcoords.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	glGenBuffers(1, &index_buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, meshes[0].indices.size() * sizeof(unsigned int), meshes[0].indices.data(), GL_STATIC_DRAW);
	index_count = meshes[0].indices.size();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
}

void My_Display()
{
    static const GLfloat gray[] = { 0.2f, 0.2f, 0.2f, 1.0f };
    static const GLfloat ones[] = { 1.0f };

    glClearBufferfv(GL_COLOR, 0, gray);
    glClearBufferfv(GL_DEPTH, 0, ones);

	float currentTime = glutGet(GLUT_ELAPSED_TIME) * 0.001f;

    glBindTexture(GL_TEXTURE_2D, tex_object[tex_index]);

    glUseProgram(program);

    mat4 proj_matrix = perspective(deg2rad(60.0f), 1.0f, 0.1f, 1000.0f);
    mat4 mv_matrix = translate(mat4(1.0f), vec3(0.0f, 0.0f, -3.0f)) *
                            rotate(mat4(1.0f), deg2rad((float)currentTime * 19.3f), vec3(0.0f, 1.0f, 0.0f)) *
                            rotate(mat4(1.0f), deg2rad((float)currentTime * 21.1f), vec3(0.0f, 0.0f, 1.0f));

    glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, &mv_matrix[0][0]);
    glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, &proj_matrix[0][0]);

	// glDrawArrays or glDrawElements
	// glDrawArrays(GL_TRIANGLES, 0, vertex_count);
	// void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void * indices);
	// 告訴 shader 要抓哪些點組成三角形，index_count 每三個會一組
    glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, 0);

	glutSwapBuffers();
}

void My_Keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'T':
		case 't':
            tex_index++;
            if (tex_index > 1)
                tex_index = 0;
			glutPostRedisplay();
            break;
    }
}

void My_Reshape(int width, int height)
{
	glViewport(0, 0, width, height);
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
	glutKeyboardFunc(My_Keyboard);
	glutTimerFunc(16, My_Timer, 0);
	///////////////////////////////

	// Enter main event loop.
	//////////////
	glutMainLoop();
	//////////////
	return 0;
}
```

# Simple Texture
```cpp
#include "../../Include/Common.h"

static const char * vs_source[] =
{
	"#version 410 core                                                              \n"
	"                                                                               \n"
	"void main(void)                                                                \n"
	"{                                                                              \n"
	"    const vec4 vertices[] = vec4[](vec4( 0.75, -0.75, 0.5, 1.0),               \n"
	"                                   vec4(-0.75, -0.75, 0.5, 1.0),               \n"
	"                                   vec4( 0.75,  0.75, 0.5, 1.0));              \n"
	"                                                                               \n"
	"    gl_Position = vertices[gl_VertexID];                                       \n"
	"}                                                                              \n"
};

static const char * fs_source[] =
{
	"#version 410 core                                                              \n"
	"                                                                               \n"
	"uniform sampler2D s;                                                           \n"
	"                                                                               \n"
	"out vec4 color;                                                                \n"
	"                                                                               \n"
	"void main(void)                                                                \n"
	"{                                                                              \n"
	"    color = texelFetch(s, ivec2(gl_FragCoord.xy) - ivec2(75, 75), 0);          \n"
	"}                                                                              \n"
};

void generate_texture(float * data, int width, int height)
{
	int x, y;

	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			data[(y * width + x) * 4 + 0] = (float)((x & y) & 0xFF) / 255.0f;
			data[(y * width + x) * 4 + 1] = (float)((x | y) & 0xFF) / 255.0f;
			data[(y * width + x) * 4 + 2] = (float)((x ^ y) & 0xFF) / 255.0f;
			data[(y * width + x) * 4 + 3] = 1.0f;
		}
	}
}

GLuint          program;
GLuint          vao;
GLuint			texture;

void My_Init()
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	// Generate a name for the texture
	glGenTextures(1, &texture);

	// Now bind it to the context using the GL_TEXTURE_2D binding point
	glBindTexture(GL_TEXTURE_2D, texture);

	// Define some data to upload into the texture
	float * data = new float[450 * 450 * 4];

	// generate_texture() is a function that fills memory with image data
	generate_texture(data, 450, 450);

	// Assume the texture is already bound to the GL_TEXTURE_2D target
	glTexImage2D(GL_TEXTURE_2D,  // 2D texture
		0,              // Level 0
		GL_RGBA,
		450, 450,       // 450 x 450 texels, replace entire image
        0,
		GL_RGBA,        // Four channel data
		GL_FLOAT,       // Floating point data
		data);          // Pointer to data
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Free the memory we allocated before - \GL now has our data
	delete [] data;

	program = glCreateProgram();
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, fs_source, NULL);
	glCompileShader(fs);

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, vs_source, NULL);
	glCompileShader(vs);

	glAttachShader(program, vs);
	glAttachShader(program, fs);
    printGLShaderLog(vs);
    printGLShaderLog(fs);

	glLinkProgram(program);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
}

// GLUT callback. Called to draw the scene.
void My_Display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	static const GLfloat green[] = { 0.0f, 0.25f, 0.0f, 1.0f };
	glClearBufferfv(GL_COLOR, 0, green);
	glUseProgram(program);

	glDrawArrays(GL_TRIANGLES, 0, 3);

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

# Single Triangle Buffer
```cpp
#include "../../Include/Common.h"

static const char * vs_source[] =
{
	"#version 410	                                                   \n"
	"                                                                  \n"
	"layout(location = 0) in vec3 iv3vertex;                           \n"
	"layout(location = 1) in vec3 iv3color;                            \n"
	"                                                                  \n"
	"out vec3 vv3color;                                                \n"
	"                                                                  \n"
	"void main(void)                                                   \n"
	"{                                                                 \n"
	"    gl_Position = vec4(iv3vertex, 1.0);                           \n"
	"    vv3color = iv3color;                                          \n"
	"}                                                                 \n"
};

static const char * fs_source[] =
{
	"#version 410		                                               \n"
	"                                                                  \n"
	"in vec3 vv3color;                                                 \n"
	"                                                                  \n"
	"layout(location = 0) out vec4 fragColor;                          \n"
	"                                                                  \n"
	"void main(void)                                                   \n"
	"{                                                                 \n"
	"    fragColor = vec4(vv3color, 1.0);                              \n"
	"}                                                                 \n"
};

GLuint program;
GLuint vao;

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

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
    
    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
	const float data[18] =
	{
		-0.5f, -0.4f, 0.0f,	//Position
		 0.5f, -0.4f, 0.0f,
		 0.0f,  0.6f, 0.0f,

		 1.0f,  0.0f, 0.0f,	//Color
		 0.0f,  1.0f, 0.0f,
		 0.0f,  0.0f, 1.0f
	};

	glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)(sizeof(float) * 9));//offset

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
}

// GLUT callback. Called to draw the scene.
void My_Display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(program);
    glDrawArrays(GL_TRIANGLES, 0, 3);
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

# Spinning Cube

```cpp
#include "../../Include/Common.h"

using namespace glm;

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
	"    vec4 color;                                                    \n"
	"} fs_in;                                                           \n"
	"                                                                   \n"
	"void main(void)                                                    \n"
	"{                                                                  \n"
	"    color = fs_in.color;                                           \n"
	"}                                                                  \n"
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

GLuint          program;
GLuint          vao;
GLuint          buffer;
GLint           mv_location;
GLint           proj_location;
mat4			 proj_matrix(1.0f);

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

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_positions), vertex_positions, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);

	mv_location = glGetUniformLocation(program, "mv_matrix");
	proj_location = glGetUniformLocation(program, "proj_matrix");
}

// GLUT callback. Called to draw the scene.
void My_Display()
{
	static const GLfloat green[] = { 0.0f, 0.25f, 0.0f, 1.0f };
	static const GLfloat one = 1.0f;
	glClearBufferfv(GL_COLOR, 0, green);
	glClearBufferfv(GL_DEPTH, 0, &one);

	glUseProgram(program);

	glUniformMatrix4fv(proj_location, 1, GL_FALSE, &proj_matrix[0][0]);

	mat4 Identy_Init(1.0f);
	float currentTime = glutGet(GLUT_ELAPSED_TIME) * 0.001f;
	mat4 mv_matrix = translate(Identy_Init, vec3(0.0f, 0.0f, -4.0f));
    mv_matrix = translate(mv_matrix, vec3(sinf(2.1f * currentTime) * 0.5f, cosf(1.7f * currentTime) * 0.5f,	sinf(1.3f * currentTime) * cosf(1.5f * currentTime) * 2.0f));
	mv_matrix = rotate(mv_matrix, deg2rad(currentTime * 45.0f), vec3(0.0f, 1.0f, 0.0f));
	mv_matrix = rotate(mv_matrix, deg2rad(currentTime * 81.0f), vec3(1.0f, 0.0f, 0.0f));
	glUniformMatrix4fv(mv_location, 1, GL_FALSE, &mv_matrix[0][0]);

	glDrawArrays(GL_TRIANGLES, 0, 36);
	glutSwapBuffers();
}

void My_Reshape(int width, int height)
{
	glViewport(0, 0, width, height);

	float viewportAspect = (float)width / (float)height;
	proj_matrix = perspective(deg2rad(50.0f), viewportAspect, 0.1f, 100.0f);
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

# Tunnel

```cpp
#include "../../Include/Common.h"

using namespace glm;

static const char * vs_source[] =
{
	"#version 410 core                                                      \n"
	"                                                                       \n"
	"out VS_OUT                                                             \n"
	"{                                                                      \n"
	"    vec2 tc;                                                           \n"
	"} vs_out;                                                              \n"
	"                                                                       \n"
	"uniform mat4 mvp;                                                      \n"
	"uniform float offset;                                                  \n"
	"                                                                       \n"
	"void main(void)                                                        \n"
	"{                                                                      \n"
	"    const vec2[4] position = vec2[4](vec2(-0.75, -0.75),               \n"
	"                                     vec2( 0.75, -0.75),               \n"
	"                                     vec2(-0.75,  0.75),               \n"
	"                                     vec2( 0.75,  0.75));              \n"
	"    vs_out.tc = (position[gl_VertexID].xy + vec2(offset, 0.5)) *       \n"
	"                vec2(30.0, 1.0);                                       \n"
	"    gl_Position = mvp * vec4(position[gl_VertexID], 0.0, 1.0);         \n"
	"}                                                                      \n"

};

static const char * fs_source[] =
{
	"#version 410 core                                                      \n"
	"                                                                       \n"
	"layout (location = 0) out vec4 color;                                  \n"
	"                                                                       \n"
	"in VS_OUT                                                              \n"
	"{                                                                      \n"
	"    vec2 tc;                                                           \n"
	"} fs_in;                                                               \n"
	"                                                                       \n"
	"uniform sampler2D tex;                                                 \n"
	"                                                                       \n"
	"void main(void)                                                        \n"
	"{                                                                      \n"
	"    color = texture(tex, fs_in.tc);                                    \n"
	"}                                                                      \n"
};

GLuint          program;
GLuint          vao;
mat4			  proj_matrix(1.0f);

struct
{
	GLint       mvp;
	GLint       offset;
} uniforms;

GLuint          tex_wall;
GLuint          tex_ceiling;
GLuint          tex_floor;

enum FilterTypes
{
	NEAREST = 1,
	LINEAR,
	LINEAR_MIPMAP,
	ANISOTROPIC
};

float maxAniso = 1.0f;
bool anisoSupport = false;

void My_Init()
{
    if (GL_EXT_texture_filter_anisotropic)
        anisoSupport = true;
	// anisoSupport = glewIsSupported("GL_EXT_texture_filter_anisotropic");
	if(anisoSupport)
	{
		glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
	}
	printf("Anisotropic Filtering is %ssupported\n", anisoSupport ? "" : "not ");
	printf("Max Anisotropy: %.1f\n", maxAniso);

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

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	uniforms.mvp = glGetUniformLocation(program, "mvp");
	uniforms.offset = glGetUniformLocation(program, "offset");

	TextureData tex = loadImg("../../Media/Textures/brick.png");
	glGenTextures(1, &tex_wall);
	glBindTexture(GL_TEXTURE_2D, tex_wall);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width, tex.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.data);
	glGenerateMipmap(GL_TEXTURE_2D);
	delete[] tex.data;

	tex = loadImg("../../Media/Textures/ceiling.png");
	glGenTextures(1, &tex_ceiling);
	glBindTexture(GL_TEXTURE_2D, tex_ceiling);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width, tex.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.data);
	glGenerateMipmap(GL_TEXTURE_2D);
	delete[] tex.data;

	tex = loadImg("../../Media/Textures/floor.png");
	glGenTextures(1, &tex_floor);
	glBindTexture(GL_TEXTURE_2D, tex_floor);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width, tex.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.data);
	glGenerateMipmap(GL_TEXTURE_2D);
	delete[] tex.data;

	int i;
	GLuint textures[] = { tex_floor, tex_wall, tex_ceiling };

	for (i = 0; i < 3; i++)
	{
		glBindTexture(GL_TEXTURE_2D, textures[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
}

// GLUT callback. Called to draw the scene.
void My_Display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	float currentTime = glutGet(GLUT_ELAPSED_TIME) * 0.001f;

	glUseProgram(program);

	glUniform1f(uniforms.offset, currentTime * 0.003f);
	mat4 Identy_Init(1.0);
	int i;
	GLuint textures[] = { tex_wall, tex_floor, tex_wall, tex_ceiling };
	for (i = 0; i < 4; i++)
	{
		mat4 mv_matrix = rotate(Identy_Init,deg2rad(90.0f * (float)i), vec3(0.0f, 0.0f, 1.0f));
		mv_matrix = translate(mv_matrix,vec3(-0.5f, 0.0f, -10.0f));
		mv_matrix = rotate(mv_matrix,deg2rad(90.0f),vec3( 0.0f, 1.0f, 0.0f));
		mv_matrix = scale(mv_matrix,vec3(50.0f, 1.0f, 1.0f));
		mat4 mvp = proj_matrix * mv_matrix;

		glUniformMatrix4fv(uniforms.mvp, 1, GL_FALSE, &mvp[0][0]);

		glBindTexture(GL_TEXTURE_2D, textures[i]);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}

	//glDrawArrays(GL_TRIANGLES, 0, 3);

	glutSwapBuffers();
}

void My_Reshape(int width, int height)
{
	glViewport(0, 0, width, height);
	
	float viewportAspect = (float)width / (float)height;
	
	proj_matrix = perspective(deg2rad(60.0f), viewportAspect, 0.1f, 1000.0f);
}

void My_Timer(int val)
{
	glutPostRedisplay();
	glutTimerFunc(16, My_Timer, val);
}

void My_Menu(int val)
{
	GLuint textures[] = { tex_floor, tex_wall, tex_ceiling };

	if(anisoSupport)
	{
		// Reset to default 1.0
		for (int i = 0; i < 3; i++)
		{
			glBindTexture(GL_TEXTURE_2D, textures[i]);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 1.0f); 
		}
	}

	if(val == NEAREST)
	{
		for (int i = 0; i < 3; i++)
		{
			glBindTexture(GL_TEXTURE_2D, textures[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		}
	}
	else if(val == LINEAR)
	{
		for (int i = 0; i < 3; i++)
		{
			glBindTexture(GL_TEXTURE_2D, textures[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
	}
	else if(val == LINEAR_MIPMAP)
	{
		for (int i = 0; i < 3; i++)
		{
			glBindTexture(GL_TEXTURE_2D, textures[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
	}
	else if(val == ANISOTROPIC && anisoSupport)
	{
		for (int i = 0; i < 3; i++)
		{
			glBindTexture(GL_TEXTURE_2D, textures[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAniso); 
		}
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

	// Create GLUT menu.
	////////////////////
	glutCreateMenu(My_Menu);
	glutAddMenuEntry("Nearest", NEAREST);
	glutAddMenuEntry("Linear", LINEAR);
	glutAddMenuEntry("Linear Mipmap", LINEAR_MIPMAP);
	glutAddMenuEntry("Anisotropic", ANISOTROPIC);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
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

# Wrapmode
```cpp
#include "../../Include/Common.h"

static const char * vs_source[] =
{
	"#version 410 core                                                              \n"
	"                                                                               \n"
	"uniform vec2 offset;                                                           \n"
	"                                                                               \n"
	"out vec2 tex_coord;                                                            \n"
	"                                                                               \n"
	"void main(void)                                                                \n"
	"{                                                                              \n"
	"    const vec4 vertices[] = vec4[](vec4(-0.45, -0.45, 0.0, 1.0),               \n"
	"                                   vec4( 0.45, -0.45, 0.0, 1.0),               \n"
	"                                   vec4(-0.45,  0.45, 0.0, 1.0),               \n"
	"                                   vec4( 0.45,  0.45, 0.0, 1.0));              \n"
	"                                                                               \n"
	"    gl_Position = vertices[gl_VertexID] + vec4(offset, 0.0, 0.0);              \n"
	"    tex_coord = vertices[gl_VertexID].xy * 3.0 + vec2(0.45 * 3);               \n"
	"}                                                                              \n"
};

static const char * fs_source[] =
{
	"#version 410 core                                                              \n"
	"                                                                               \n"
	"uniform sampler2D s;                                                           \n"
	"                                                                               \n"
	"out vec4 color;                                                                \n"
	"                                                                               \n"
	"in vec2 tex_coord;                                                             \n"
	"                                                                               \n"
	"void main(void)                                                                \n"
	"{                                                                              \n"
	"    color = texture(s, tex_coord);                                             \n"
	"}                                                                              \n"
};

GLuint          program;
GLuint          vao;
GLuint          texture;

struct
{
	GLint       mvp;
	GLint       offset;
} uniforms;

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

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenTextures(1, &texture);

	// Load texture from file
	TextureData tex = loadImg("../../Media/Textures/rightarrows.png");
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width, tex.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	delete[] tex.data;

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

// GLUT callback. Called to draw the scene.
void My_Display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	static const GLenum wrapmodes[] = { GL_CLAMP_TO_EDGE, GL_REPEAT, GL_CLAMP_TO_BORDER, GL_MIRRORED_REPEAT };
	static const GLfloat yellow[] = { 0.4f, 0.4f, 0.0f, 1.0f };

	float currentTime = glutGet(GLUT_ELAPSED_TIME) * 0.001f;

	glUseProgram(program);
	static const float offsets[] =
	{
		-0.5f, -0.5f,
		 0.5f, -0.5f,
		-0.5f,  0.5f,
		 0.5f,  0.5f
	};

	GLint loc_offset;
	loc_offset = glGetUniformLocation(program, "offset");

	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, yellow);

	for (int i = 0; i < 4; i++)
	{
		glUniform2fv(loc_offset, 1, &offsets[i * 2]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapmodes[i]); // 水平
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapmodes[i]); // 垂直

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	}

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