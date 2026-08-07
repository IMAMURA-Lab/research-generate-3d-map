// Crouch.cs
// このスクリプトは「しゃがみ（Crouch）」の挙動を担当します。
// ・キー入力によるしゃがみ判定
// ・移動速度の低下
// ・カメラ（頭）の高さ変更
// ・カプセルコライダーの高さ変更
// をまとめて制御しています。

using UnityEngine;

public class Crouch : MonoBehaviour
{
    // しゃがみ操作に使うキー（デフォルト：左Ctrl）
    public KeyCode key = KeyCode.LeftControl;

    [Header("Slow Movement")]
    [Tooltip("Movement to slow down when crouched.")]
    // 移動処理を担当する FirstPersonMovement
    public FirstPersonMovement movement;

    [Tooltip("Movement speed when crouched.")]
    // しゃがみ中の移動速度
    public float movementSpeed = 2;

    [Header("Low Head")]
    [Tooltip("Head to lower when crouched.")]
    // しゃがみ時に下げる「頭」（通常はカメラ）
    public Transform headToLower;

    [HideInInspector]
    // 通常時の頭の Y ローカル座標（初回のみ保存）
    public float? defaultHeadYLocalPosition;

    // しゃがみ時の頭の Y ローカル座標
    public float crouchYHeadPosition = 1;

    [Tooltip("Collider to lower when crouched.")]
    // しゃがみ時に高さを変更する CapsuleCollider
    public CapsuleCollider colliderToLower;

    [HideInInspector]
    // 通常時のコライダーの高さ（初回のみ保存）
    public float? defaultColliderHeight;

    // 現在しゃがんでいるかどうか
    public bool IsCrouched { get; private set; }

    // しゃがみ開始・終了時に外部へ通知するイベント
    public event System.Action CrouchStart, CrouchEnd;


    void Reset()
    {
        // 自動的に関連コンポーネントを探して設定する
        movement = GetComponentInParent<FirstPersonMovement>();
        headToLower = movement.GetComponentInChildren<Camera>().transform;
        colliderToLower = movement.GetComponentInChildren<CapsuleCollider>();
    }

    void LateUpdate()
    {
        // しゃがみキーが押されている間の処理
        if (Input.GetKey(key))
        {
            // ===== カメラ（頭）を下げる処理 =====
            if (headToLower)
            {
                // 通常時の頭の高さが未保存ならここで保存
                if (!defaultHeadYLocalPosition.HasValue)
                {
                    defaultHeadYLocalPosition = headToLower.localPosition.y;
                }

                // 頭の高さをしゃがみ位置に変更
                headToLower.localPosition = new Vector3(
                    headToLower.localPosition.x,
                    crouchYHeadPosition,
                    headToLower.localPosition.z
                );
            }

            // ===== コライダーを低くする処理 =====
            if (colliderToLower)
            {
                // 通常時のコライダー高さを未保存ならここで保存
                if (!defaultColliderHeight.HasValue)
                {
                    defaultColliderHeight = colliderToLower.height;
                }

                // コライダーをどれだけ下げるか計算
                float loweringAmount;
                if (defaultHeadYLocalPosition.HasValue)
                {
                    // 頭の下げ量に合わせる
                    loweringAmount = defaultHeadYLocalPosition.Value - crouchYHeadPosition;
                }
                else
                {
                    // 頭が無い場合はコライダー高さの半分
                    loweringAmount = defaultColliderHeight.Value * .5f;
                }

                // コライダーの高さを縮小（0未満にならないよう制限）
                colliderToLower.height = Mathf.Max(
                    defaultColliderHeight.Value - loweringAmount,
                    0
                );

                // コライダーの中心を調整（地面に食い込まないように）
                colliderToLower.center = Vector3.up * colliderToLower.height * .5f;
            }

            // ===== 状態管理 =====
            if (!IsCrouched)
            {
                // 初めてしゃがんだ瞬間のみ実行
                IsCrouched = true;

                // 移動速度をしゃがみ用に上書き
                SetSpeedOverrideActive(true);

                // 外部へ「しゃがみ開始」を通知
                CrouchStart?.Invoke();
            }
        }
        else
        {
            // しゃがみキーが離されたとき
            if (IsCrouched)
            {
                // ===== 頭を元の高さに戻す =====
                if (headToLower)
                {
                    headToLower.localPosition = new Vector3(
                        headToLower.localPosition.x,
                        defaultHeadYLocalPosition.Value,
                        headToLower.localPosition.z
                    );
                }

                // ===== コライダーを元に戻す =====
                if (colliderToLower)
                {
                    colliderToLower.height = defaultColliderHeight.Value;
                    colliderToLower.center = Vector3.up * colliderToLower.height * .5f;
                }

                // ===== 状態リセット =====
                IsCrouched = false;

                // 移動速度の上書きを解除
                SetSpeedOverrideActive(false);

                // 外部へ「しゃがみ終了」を通知
                CrouchEnd?.Invoke();
            }
        }
    }


    #region Speed override.
    // FirstPersonMovement の速度上書き機構を操作する
    void SetSpeedOverrideActive(bool state)
    {
        // movement が無ければ何もしない
        if (!movement)
        {
            return;
        }

        if (state)
        {
            // しゃがみ時：速度上書き関数を追加
            if (!movement.speedOverrides.Contains(SpeedOverride))
            {
                movement.speedOverrides.Add(SpeedOverride);
            }
        }
        else
        {
            // 立ち状態：速度上書き関数を削除
            if (movement.speedOverrides.Contains(SpeedOverride))
            {
                movement.speedOverrides.Remove(SpeedOverride);
            }
        }
    }

    // speedOverrides に登録される関数
    // 呼ばれるとしゃがみ中の移動速度を返す
    float SpeedOverride() => movementSpeed;
    #endregion
}
