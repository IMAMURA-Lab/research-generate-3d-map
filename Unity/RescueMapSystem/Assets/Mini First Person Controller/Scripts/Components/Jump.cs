// Jump.cs
// プレイヤーのジャンプ処理を担当するスクリプト
// ・入力（Jumpボタン）を監視
// ・地面に接地している場合のみジャンプを許可
// ・ジャンプ時にイベントを発行（音など他処理用）

using UnityEngine;

public class Jump : MonoBehaviour
{
    // プレイヤーに付いている Rigidbody（物理的な力を加えるため）
    Rigidbody rigidbody;

    // ジャンプの強さ（Inspectorから調整可能）
    public float jumpStrength = 2;

    // ジャンプが発生したことを通知するイベント
    // FirstPersonAudio などがこのイベントを購読する
    public event System.Action Jumped;

    // 接地判定用スクリプト
    // SerializeField により private でも Inspector に表示される
    [SerializeField, Tooltip("Prevents jumping when the transform is in mid-air.")]
    GroundCheck groundCheck;

    void Reset()
    {
        // コンポーネントが追加・リセットされたときに自動で呼ばれる
        // 子オブジェクトから GroundCheck を探して設定する
        groundCheck = GetComponentInChildren<GroundCheck>();
    }

    void Awake()
    {
        // オブジェクト生成時に Rigidbody を取得
        // ジャンプ時に AddForce を使うため必須
        rigidbody = GetComponent<Rigidbody>();
    }

    void LateUpdate()
    {
        // 毎フレーム「Jump」ボタンが押された瞬間を検出
        // かつ、
        // ・groundCheck が存在しない（＝接地判定を使わない）
        // または
        // ・groundCheck があり、現在地面に接地している
        if (Input.GetButtonDown("Jump") && (!groundCheck || groundCheck.isGrounded))
        {
            // 上方向（Vector3.up）に力を加えてジャンプさせる
            // 100 はスケール調整用の固定倍率
            rigidbody.AddForce(Vector3.up * 100 * jumpStrength);

            // ジャンプが発生したことをイベントで通知
            // （ジャンプ音再生など、他スクリプトが反応する）
            Jumped?.Invoke();
        }
    }
}
