from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from threading import Lock
import numpy as np
import sys
import array
import math
import ctypes
import pyzed.sl as sl

# 円周率（簡易定義）
M_PI = 3.1415926

# 以下は GLSL シェーダーのソースコード（文字列）です。
# 各シェーダーは OpenGL に渡してコンパイル・リンクされ、頂点変換や色計算に使われます。
MESH_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_Vertex;
uniform mat4 u_mvpMatrix;
uniform vec3 u_color;
out vec3 b_color;
void main() {
    b_color = u_color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

FPC_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec4 in_Vertex;
uniform mat4 u_mvpMatrix;
uniform vec3 u_color;
out vec3 b_color;
void main() {
   b_color = u_color;
   gl_Position = u_mvpMatrix * vec4(in_Vertex.xyz, 1);
}
"""

VERTEX_SHADER = """
# version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in vec4 in_Color;
uniform mat4 u_mvpMatrix;
out vec4 b_color;
void main() {
    b_color = in_Color;
    gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 b_color;
layout(location = 0) out vec4 color;
void main() {
   color = vec4(b_color,1);
}
"""

# シェーダープログラム（頂点＋フラグメント）を作成・管理するクラス
class Shader:
    def __init__(self, _vs, _fs):

        # プログラムオブジェクトを作成
        self.program_id = glCreateProgram()
        # 頂点シェーダ／フラグメントシェーダをコンパイル
        vertex_id = self.compile(GL_VERTEX_SHADER, _vs)
        fragment_id = self.compile(GL_FRAGMENT_SHADER, _fs)

        # コンパイル済みシェーダをプログラムにアタッチ
        glAttachShader(self.program_id, vertex_id)
        glAttachShader(self.program_id, fragment_id)
        # 頂点属性の位置を明示的にバインド（ただし属性名はコード内で"in_vertex"/"in_texCoord"を想定）
        glBindAttribLocation( self.program_id, 0, "in_vertex")
        glBindAttribLocation( self.program_id, 1, "in_texCoord")
        # プログラムのリンク
        glLinkProgram(self.program_id)

        # リンクエラー時のハンドリング
        if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.program_id)
            glDeleteProgram(self.program_id)
            glDeleteShader(vertex_id)
            glDeleteShader(fragment_id)
            raise RuntimeError('Error linking program: %s' % (info))
        # シェーダオブジェクトはリンク後に削除してよい
        glDeleteShader(vertex_id)
        glDeleteShader(fragment_id)

    # シェーダをコンパイルして ID を返す
    def compile(self, _type, _src):
        try:
            shader_id = glCreateShader(_type)
            if shader_id == 0:
                print("ERROR: shader type {0} does not exist".format(_type))
                exit()

            # ソースを設定してコンパイル
            glShaderSource(shader_id, _src)
            glCompileShader(shader_id)
            # コンパイルチェック
            if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                info = glGetShaderInfoLog(shader_id)
                glDeleteShader(shader_id)
                raise RuntimeError('Shader compilation failed: %s' % (info))
            return shader_id
        except:
            # 何か失敗したら確実にシェーダを削除して例外を再送出
            glDeleteShader(shader_id)
            raise

    # コンパイル・リンク済みプログラムの ID を取得するユーティリティ
    def get_program_id(self):
        return self.program_id

# 画像を描画するためのフラグメント／頂点シェーダ（テクスチャ描画用）
IMAGE_FRAGMENT_SHADER = """
#version 330 core
in vec2 UV;
out vec4 color;
uniform sampler2D texImage;
uniform bool revert;
uniform bool rgbflip;
void main() {
    vec2 scaler  =revert?vec2(UV.x,1.f - UV.y):vec2(UV.x,UV.y);
    vec3 rgbcolor = rgbflip?vec3(texture(texImage, scaler).zyx):vec3(texture(texImage, scaler).xyz);
    color = vec4(rgbcolor,1);
}
"""

IMAGE_VERTEX_SHADER = """
#version 330
layout(location = 0) in vec3 vert;
out vec2 UV;
void main() {
    UV = (vert.xy+vec2(1,1))/2;
    gl_Position = vec4(vert, 1);
}
"""

# OpenGL テクスチャに ZED カメラの画像を転送・描画するためのハンドラ
class ImageHandler:
    """
    Class that manages the image stream to render with OpenGL
    """
    def __init__(self):
        # テクスチャ ID・バッファ等の初期値
        self.tex_id = 0
        self.image_tex = 0
        self.quad_vb = 0
        self.is_called = 0

    # リソース解放（ここでは単純に ID をクリア）
    def close(self):
        if self.image_tex:
            self.image_tex = 0

    # 初期化: テクスチャと描画用 VBO を生成
    def initialize(self, _res):    
        # イメージ用のシェーダを生成
        self.shader_image = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER)
        # シェーダ内のテクスチャユニフォームの位置を取得
        self.tex_id = glGetUniformLocation( self.shader_image.get_program_id(), "texImage")

        # 全画面に描画する三角形 2 個分（矩形）を用意
        g_quad_vertex_buffer_data = np.array([-1, -1, 0,
                                                1, -1, 0,
                                                -1, 1, 0,
                                                -1, 1, 0,
                                                1, -1, 0,
                                                1, 1, 0], np.float32)

        # VBO を生成してデータを転送
        self.quad_vb = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vb)
        glBufferData(GL_ARRAY_BUFFER, g_quad_vertex_buffer_data.nbytes,
                     g_quad_vertex_buffer_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # テクスチャの準備
        glEnable(GL_TEXTURE_2D)

        # テクスチャ名を生成
        self.image_tex = glGenTextures(1)
        
        # 生成したテクスチャをバインド
        glBindTexture(GL_TEXTURE_2D, self.image_tex)
        
        # フィルタ設定（拡大縮小）
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # テクスチャ領域を確保（まだデータは入れない）
        # None を渡すことでメモリを確保するが初期値は未定義
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _res.width, _res.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        
        # テクスチャのバインド解除
        glBindTexture(GL_TEXTURE_2D, 0)   

    # ZED の sl.Mat 等からピクセルデータを読み取り、既存テクスチャに部分更新する
    def push_new_image(self, _zed_mat):
        glBindTexture(GL_TEXTURE_2D, self.image_tex)
        # ZED の Mat のポインタを直接渡してサブイメージ更新（高速）
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _zed_mat.get_width(), _zed_mat.get_height(), GL_RGBA, GL_UNSIGNED_BYTE,  ctypes.c_void_p(_zed_mat.get_pointer()))
        glBindTexture(GL_TEXTURE_2D, 0)            

    # 登録したシェーダとテクスチャで quad を描画
    def draw(self):
        glUseProgram(self.shader_image.get_program_id())
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.image_tex)
        glUniform1i(self.tex_id, 0)

        # Y 軸反転や RGB -> BGR 反転を行うためのフラグをシェーダに渡す
        glUniform1i(glGetUniformLocation(self.shader_image.get_program_id(), "revert"), 1)
        glUniform1i(glGetUniformLocation(self.shader_image.get_program_id(), "rgbflip"), 1)

        # 頂点配列を有効化して VBO から座標を読み取る
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vb)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDisableVertexAttribArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)            
        glUseProgram(0)

# OpenGL 描画のメインクラス。ZED の pose とマッピング結果を合成して表示する
class GLViewer:
    """
    Class that manages the rendering in OpenGL
    """
    def __init__(self):
        # 初期状態のフラグやロック
        self.available = False
        self.mutex = Lock()
        # 描画モード（メッシュ描画か FPC 描画か）
        self.draw_mesh = False
        self.new_chunks = False
        self.chunks_pushed = False
        self.change_state = False
        # プロジェクション行列（sl.Matrix4f を使用）
        self.projection = sl.Matrix4f()
        self.projection.set_identity()
        # クリップ平面
        self.znear = 0.5
        self.zfar = 100.
        # 画像ハンドラ
        self.image_handler = ImageHandler()
        # 部分マップ（チャンク）を格納する配列
        self.sub_maps = []
        # カメラの姿勢（sl.Transform）
        self.pose = sl.Transform().set_identity()
        # トラッキング・マッピングの状態
        self.tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
        self.mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

    # ウィンドウと OpenGL の初期化、シェーダ生成、コールバック登録等を行う
    def init(self, _params, _mesh, _create_mesh): 
        # GLUT の初期化
        glutInit()
        wnd_w = glutGet(GLUT_SCREEN_WIDTH)
        wnd_h = glutGet(GLUT_SCREEN_HEIGHT)
        width = wnd_w*0.9
        height = wnd_h*0.9
     
        # ZED カメラ解像度より大きすぎる場合は制限
        if width > _params.image_size.width and height > _params.image_size.height:
            width = _params.image_size.width
            height = _params.image_size.height

        # ウィンドウサイズ・位置・モード設定
        glutInitWindowSize(int(width), int(height))
        glutInitWindowPosition(0, 0) # The window opens at the upper left corner of the screen
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_SRGB)
        glutCreateWindow(b"ZED Spatial Mapping")
        glViewport(0, 0, int(width), int(height))

        # ウィンドウを閉じてもプログラムは継続する設定
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT_ACTION_CONTINUE_EXECUTION)

        # ラインのアンチエイリアス設定
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # 画像描画用の初期化（テクスチャ等）
        self.image_handler.initialize(_params.image_size)

        # メッシュ（または FPC: first point cloud のようなもの）の初期化
        self.init_mesh(_mesh, _create_mesh)

        # メッシュを描画するかどうかで使うシェーダを切り替える
        if(self.draw_mesh):
            self.shader_image = Shader(MESH_VERTEX_SHADER, FRAGMENT_SHADER)
        else:
            self.shader_image = Shader(FPC_VERTEX_SHADER, FRAGMENT_SHADER)

        # シェーダ内のユニフォーム変数のロケーションを取得
        self.shader_MVP = glGetUniformLocation(self.shader_image.get_program_id(), "u_mvpMatrix")
        self.shader_color_loc = glGetUniformLocation(self.shader_image.get_program_id(), "u_color")
        # カメラの射影行列を設定
        self.set_render_camera_projection(_params)

        # 表示のための線幅やポイントサイズを設定
        glLineWidth(1.)
        glPointSize(4.)

        # GLUT のコールバックを登録
        glutDisplayFunc(self.draw_callback)
        glutIdleFunc(self.idle)   
        glutKeyboardUpFunc(self.keyReleasedCallback)
        glutCloseFunc(self.close_func)

        # 内部フラグ初期化
        self.ask_clear = False
        self.available = True

        # ワイヤーフレームの色（頂点ではなく均一色）
        self.vertices_color = [0.12,0.53,0.84] 
        
        # 初期状態ではチャンクがプッシュ済み（描画可能）
        self.chunks_pushed = True

    # メッシュ（または FPC）オブジェクトの受け取り・設定
    def init_mesh(self, _mesh, _create_mesh):
        self.draw_mesh = _create_mesh
        self.mesh = _mesh

    # ZED カメラパラメータから射影行列を作成（簡易的な処理）
    def set_render_camera_projection(self, _params):
        # 少し FOV を大きめにとることで黒い境界を作る
        fov_y = (_params.v_fov + 0.5) * M_PI / 180
        fov_x = (_params.h_fov + 0.5) * M_PI / 180

        # 射影行列の主要要素を設定（対称射影）
        self.projection[(0,0)] = 1. / math.tan(fov_x * .5)
        self.projection[(1,1)] = 1. / math.tan(fov_y * .5)
        self.projection[(2,2)] = -(self.zfar + self.znear) / (self.zfar - self.znear)
        self.projection[(3,2)] = -1.
        self.projection[(2,3)] = -(2. * self.zfar * self.znear) / (self.zfar - self.znear)
        self.projection[(3,3)] = 0.
    
    # GL 上にテキストを描画するユーティリティ（簡易）
    def print_GL(self, _x, _y, _string):
        glRasterPos(_x, _y)
        for i in range(len(_string)):
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ctypes.c_int(ord(_string[i])))

    # 利用可能かどうか（メインループの処理を行う）
    def is_available(self):
        if self.available:
            glutMainLoopEvent()
        return self.available

    # オブジェクトの描画可否（ここではトラッキング状態に基づく単純な判定）
    def render_object(self, _object_data):      # _object_data of type sl.ObjectData
        if _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK or _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF:
            return True
        else:
            return False

    # 外部からチャンクが更新されたことを通知（次の update で反映される）
    def update_chunks(self):
        self.new_chunks = True
        self.chunks_pushed = False
    
    # チャンクが描画用に準備済みかどうかを返す
    def chunks_updated(self):
        return self.chunks_pushed

    # 現在のメッシュをクリアする要求を出す
    def clear_current_mesh(self):
        self.ask_clear = True
        self.new_chunks = True

    # 描画するビューの更新（画像・姿勢・状態を受け取り内部で保持）
    def update_view(self, _image, _pose, _tracking_state, _mapping_state):     
        self.mutex.acquire()
        if self.available:
            # 画像テクスチャを更新
            self.image_handler.push_new_image(_image)
            # 現在の pose / 状態を保存
            self.pose = _pose
            self.tracking_state = _tracking_state
            self.mapping_state = _mapping_state
        self.mutex.release()
        copy_state = self.change_state
        self.change_state = False
        return copy_state

    # アイドル時に再描画をリクエスト
    def idle(self):
        if self.available:
            glutPostRedisplay()

    # 外部から終了要求が来たときに呼ばれる
    def exit(self):      
        if self.available:
            self.available = False
            self.image_handler.close()

    # ウィンドウクローズ時の処理（リソース解放）
    def close_func(self): 
        if self.available:
            self.available = False
            self.image_handler.close()      

    # キーイベントハンドラ（'q' や ESC で終了、スペースで状態を切り替え）
    def keyReleasedCallback(self, key, x, y):
        if ord(key) == 113 or ord(key) == 27:   # 'q' key または ESC
            self.close_func()
        if  ord(key) == 32:                     # space bar
            self.change_state = True

    # GLUT の描画コールバック
    def draw_callback(self):
        if self.available:
            # バッファクリア
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0, 0, 0, 1.0)

            # 排他制御して描画更新
            self.mutex.acquire()
            self.update()
            self.draw()
            self.print_text()
            self.mutex.release()  

            # ダブルバッファを入れ替え
            glutSwapBuffers()
            glutPostRedisplay()

    # メッシュや FPC の更新処理。new_chunks フラグが立っていればチャンクを取り込む
    def update(self):
        if self.new_chunks:
            if self.ask_clear:
                # クリア要求があれば既存の sub_maps を消す
                self.sub_maps = []
                self.ask_clear = False
            
            nb_c = len(self.mesh.chunks)

            # sub_maps の要素数をメッシュチャンク数に合わせて拡張
            if nb_c > len(self.sub_maps): 
                for n in range(len(self.sub_maps),nb_c):
                    self.sub_maps.append(SubMapObj())
            
            # 各チャンクに対して更新があれば SubMapObj に転送
            for m in range(len(self.sub_maps)):
                if (m < nb_c) and self.mesh.chunks[m].has_been_updated:
                    if self.draw_mesh:
                        # 三角形メッシュとして更新
                        self.sub_maps[m].update_mesh(self.mesh.chunks[m])
                    else:
                        # FPC（ポイントクラウド的描画）として更新
                        self.sub_maps[m].update_fpc(self.mesh.chunks[m])
                        
            # フラグをクリア
            self.new_chunks = False
            self.chunks_pushed = True

    # 実際の描画処理（画像 → メッシュオーバーレイ）
    def draw(self):  
        if self.available:
            # まずは背景（カメラ画像）を描画
            self.image_handler.draw()

            # Positional tracking が OK であればメッシュをオーバーレイして表示
            if self.tracking_state == sl.POSITIONAL_TRACKING_STATE.OK and len(self.sub_maps) > 0:
                # ワイヤーフレーム（ポリゴンは線で描画）
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

                # カメラ姿勢（pose）を射影行列に組み込む
                tmp = self.pose
                tmp.inverse()
                proj = (self.projection * tmp).m
                vpMat = proj.flatten()
                
                # シェーダを使って MVP と色を設定
                glUseProgram(self.shader_image.get_program_id())
                glUniformMatrix4fv(self.shader_MVP, 1, GL_TRUE, (GLfloat * len(vpMat))(*vpMat))
                glUniform3fv(self.shader_color_loc, 1, (GLfloat * len(self.vertices_color))(*self.vertices_color))
        
                # 各 sub_map を描画
                for m in range(len(self.sub_maps)):
                    self.sub_maps[m].draw(self.draw_mesh)

                # シェーダの使用を解除してポリゴンモードを戻す
                glUseProgram(0)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    # 画面にテキストで状態を表示する処理
    def print_text(self):
        if self.available:
            # キー操作のヘルプや状態表示（描画色を切り替えて表示）
            if self.mapping_state == sl.SPATIAL_MAPPING_STATE.NOT_ENABLED:
                glColor3f(0.15, 0.15, 0.15)
                # self.print_GL(-0.99, 0.9, "Hit Space Bar to activate Spatial Mapping.")
            else:
                glColor3f(0.25, 0.25, 0.25)
                # self.print_GL(-0.99, 0.9, "Hit Space Bar to stop spatial mapping.")

            positional_tracking_state_str = "POSITIONAL TRACKING STATE : "
            spatial_mapping_state_str = "SPATIAL MAPPING STATE : "
            state_str = ""

            # トラッキング／マッピングの状態に応じて色と表示文字列を変更
            if self.tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                if self.mapping_state == sl.SPATIAL_MAPPING_STATE.OK or self.mapping_state == sl.SPATIAL_MAPPING_STATE.INITIALIZING:
                    glColor3f(0.25, 0.99, 0.25)
                elif self.mapping_state == sl.SPATIAL_MAPPING_STATE.NOT_ENABLED:
                    glColor3f(0.55, 0.65, 0.55)
                else:
                    glColor3f(0.95, 0.25, 0.25)
                state_str = spatial_mapping_state_str + str(self.mapping_state)
            else:
                if self.mapping_state != sl.SPATIAL_MAPPING_STATE.NOT_ENABLED:
                    glColor3f(0.95, 0.25, 0.25)
                    state_str = positional_tracking_state_str + str(self.tracking_state)
                else:
                    glColor3f(0.55, 0.65, 0.55)
                    state_str = spatial_mapping_state_str + str(sl.SPATIAL_MAPPING_STATE.NOT_ENABLED)
            # 実際に画面に描画
            self.print_GL(-0.99, 0.9, state_str)

# 部分マップ（チャンク）を表すクラス
class SubMapObj:
    def __init__(self):
        # 現在の頂点数 / VBO ID / インデックス配列等を初期化
        self.current_fc = 0
        self.vboID = None
        self.index = []         # FPC（ポイント）用のインデックス
        self.vert = []
        self.tri = []

    # 三角形メッシュ（vertices, triangles）を VBO にアップロードする
    def update_mesh(self, _chunk): 
        if(self.vboID is None):
            # VBO を頂点用とインデックス用に 2 つ生成
            self.vboID = glGenBuffers(2)

        if len(_chunk.vertices):
            # NumPy 配列を 1 次元化して転送（GLfloat 配列として）
            self.vert = _chunk.vertices.flatten()      # transform _chunk.vertices into 1D array 
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(self.vert) * self.vert.itemsize, (GLfloat * len(self.vert))(*self.vert), GL_DYNAMIC_DRAW)
        
        if len(_chunk.triangles):
            # 三角形インデックスも 1 次元配列にして ELEMENT_ARRAY_BUFFER に転送
            self.tri = _chunk.triangles.flatten()      # transform _chunk.triangles into 1D array 
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.tri) * self.tri.itemsize , (GLuint * len(self.tri))(*self.tri), GL_DYNAMIC_DRAW)
            # current_fc は描画要素数（インデックス数）
            self.current_fc = len(self.tri)

    # FPC（ポイントクラウド的描画）用に頂点とインデックスを作成して VBO に登録する
    def update_fpc(self, _chunk): 
        if(self.vboID is None):
            self.vboID = glGenBuffers(2)

        if len(_chunk.vertices):
            # 頂点配列を転送
            self.vert = _chunk.vertices.flatten()      # transform _chunk.vertices into 1D array 
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            glBufferData(GL_ARRAY_BUFFER, len(self.vert) * self.vert.itemsize, (GLfloat * len(self.vert))(*self.vert), GL_DYNAMIC_DRAW)

            # 単純に 0..N-1 のインデックスを作る
            for i in range(len(_chunk.vertices)):
                self.index.append(i)
            
            index_np = np.array(self.index)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[1])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(index_np) * index_np.itemsize, (GLuint * len(index_np))(*index_np), GL_DYNAMIC_DRAW)
            # current_fc はポイント数
            self.current_fc = len(index_np)

    # 実際に VBO の内容を描画する
    def draw(self, _draw_mesh): 
        if self.current_fc:
            # 頂点属性配列を有効化して頂点バッファをバインド
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, self.vboID[0])
            # メッシュなら 3 成分 float（x,y,z）、FPC（ここでは in_Vertex が vec4 なので 4）なら 4 成分として扱う
            if _draw_mesh == True:
                glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)
            else:
                glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,None)

            # インデックスバッファをバインドして描画
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboID[1])
            # インデックス配列（index がある場合はポイント描画、なければ三角形描画）
            if len(self.index) > 0:
                # ポイント描画（FPC）
                glDrawElements(GL_POINTS, self.current_fc, GL_UNSIGNED_INT, None)      
            else:
                # 三角形描画（メッシュ）
                glDrawElements(GL_TRIANGLES, self.current_fc, GL_UNSIGNED_INT, None)      

            # 後片付け：頂点属性を無効化
            glDisableVertexAttribArray(0)
