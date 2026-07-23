// FirstPersonLook.cs
// このスクリプトは、First Person Controller の「視点操作」を担当します。
// マウスの動きに応じてカメラ（プレイヤーの目）やキャラクター本体を回転させる役割です。

using UnityEngine;

public class FirstPersonLook : MonoBehaviour
{
    // カメラの回転対象となるキャラクター本体
    [SerializeField]
    Transform character;

    // マウス感度（横・縦の回転スピード）
    public float sensitivity = 2;

    // 回転の滑らかさ（大きいほどスムーズに動く）
    public float smoothing = 1.5f;

    // 累積回転量（カメラとキャラクター全体の回転を追跡）
    Vector2 velocity;

    // フレーム単位の回転量（滑らかにするために一時的に保持）
    Vector2 frameVelocity;


    void Reset()
    {
        // このコンポーネントがアタッチされているオブジェクトの親にある
        // FirstPersonMovement スクリプトの Transform を取得して character に設定
        character = GetComponentInParent<FirstPersonMovement>().transform;
    }

    void Start()
    {
        // マウスカーソルをゲーム画面内に固定して非表示にする
        // これにより、FPS操作中にカーソルが画面外に出なくなる
        Cursor.lockState = CursorLockMode.Locked;
    }

    void Update()
    {
        // マウスのフレームごとの差分を取得（X:横回転、Y:縦回転）
        Vector2 mouseDelta = new Vector2(Input.GetAxisRaw("Mouse X"), Input.GetAxisRaw("Mouse Y"));

        // 感度を掛けて回転量をスケーリング
        Vector2 rawFrameVelocity = Vector2.Scale(mouseDelta, Vector2.one * sensitivity);

        // 前フレームの回転量との補間で滑らかにする
        // smoothing が大きいほど動きがスムーズ（遅れる感じ）
        frameVelocity = Vector2.Lerp(frameVelocity, rawFrameVelocity, 1 / smoothing);

        // 累積回転量に加算
        velocity += frameVelocity;

        // カメラの上下回転を-90°〜90°に制限
        velocity.y = Mathf.Clamp(velocity.y, -90, 90);

        // 実際の回転処理
        // カメラ（このスクリプトがアタッチされているオブジェクト）の上下回転
        transform.localRotation = Quaternion.AngleAxis(-velocity.y, Vector3.right);

        // キャラクター本体の左右回転
        character.localRotation = Quaternion.AngleAxis(velocity.x, Vector3.up);
    }
}
