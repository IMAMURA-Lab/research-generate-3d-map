// Zoom.cs
// カメラのズーム（FOV変更）を担当するスクリプト
// ・マウスホイール入力でズームイン／アウト
// ・Field of View（視野角）を補間して自然なズームを実現
// ・ExecuteInEditMode により、エディタ上でも挙動を確認可能

using UnityEngine;

[ExecuteInEditMode]
// プレイモードでなくても Update が実行される属性
// FOV の調整結果を Scene / Inspector 上で即時確認できる
public class Zoom : MonoBehaviour
{
    // この GameObject に付いている Camera コンポーネント
    Camera camera;

    // 通常時の視野角（FOV）
    // Awake 時に Camera.fieldOfView から自動取得される
    public float defaultFOV = 60;

    // 最大ズーム時の視野角（小さいほど望遠）
    public float maxZoomFOV = 15;

    // 現在のズーム量（0～1）
    // 0 = 通常視野、1 = 最大ズーム
    [Range(0, 1)]
    public float currentZoom;

    // マウスホイール入力に対するズーム感度
    public float sensitivity = 1;

    void Awake()
    {
        // この GameObject に付いている Camera を取得
        camera = GetComponent<Camera>();

        // Camera が存在する場合
        if (camera)
        {
            // 現在のカメラFOVを「通常時のFOV」として保存
            // Inspector での初期値より Camera の設定を優先する設計
            defaultFOV = camera.fieldOfView;
        }
    }

    void Update()
    {
        // マウスホイールの入力を取得
        // mouseScrollDelta.y はホイールの回転量（前後）
        // sensitivity と 0.05f によりズーム速度を調整
        currentZoom += Input.mouseScrollDelta.y * sensitivity * .05f;

        // currentZoom を 0～1 の範囲に制限
        currentZoom = Mathf.Clamp01(currentZoom);

        // defaultFOV（通常）と maxZoomFOV（最大ズーム）を
        // currentZoom の値に応じて線形補間（Lerp）する
        camera.fieldOfView = Mathf.Lerp(defaultFOV, maxZoomFOV, currentZoom);
    }
}
