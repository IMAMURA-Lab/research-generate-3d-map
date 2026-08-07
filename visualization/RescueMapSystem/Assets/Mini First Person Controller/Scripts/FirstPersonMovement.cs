// FirstPersonMovement.cs
// このスクリプトは、First Person Controller の「移動」を担当します。
// プレイヤーの前後左右移動や、走る速度の制御を Rigidbody を通じて行います。

using System.Collections.Generic;
using UnityEngine;

public class FirstPersonMovement : MonoBehaviour
{
    // 基本の歩行速度（m/s）
    public float speed = 5;

    [Header("Running")]
    // 走る機能を有効にするかどうか
    public bool canRun = true;

    // 現在走っているかどうか（外部から参照可能だが、設定は内部でのみ行う）
    public bool IsRunning { get; private set; }

    // 走る時の速度
    public float runSpeed = 9;

    // 走るキー（デフォルトは左Shift）
    public KeyCode runningKey = KeyCode.LeftShift;

    // このプレイヤーにアタッチされている Rigidbody
    Rigidbody rigidbody;

    /// <summary>
    /// 移動速度を上書きする関数のリスト
    /// 最後に追加されたものが優先される
    /// 外部から追加可能（例: 効果アイテムで速度変化）
    /// </summary>
    public List<System.Func<float>> speedOverrides = new List<System.Func<float>>();



    void Awake()
    {
        // 自身の Rigidbody を取得
        rigidbody = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        // 毎フレーム、走っているか判定
        IsRunning = canRun && Input.GetKey(runningKey);

        // 基本速度は歩行か走行かで切り替える
        float targetMovingSpeed = IsRunning ? runSpeed : speed;

        // speedOverrides が存在する場合は最後に追加された関数の値で上書き
        if (speedOverrides.Count > 0)
        {
            targetMovingSpeed = speedOverrides[speedOverrides.Count - 1]();
        }

        // 入力を取得してターゲット速度を決定
        // Horizontal: A/D, 左右矢印
        // Vertical: W/S, 上下矢印
        Vector2 targetVelocity = new Vector2(
            Input.GetAxis("Horizontal") * targetMovingSpeed,
            Input.GetAxis("Vertical") * targetMovingSpeed
        );

        // Rigidbody に速度を設定して移動させる
        // transform.rotation を掛けることで、キャラクターの向きに沿った移動になる
        // Y 軸は現在の Rigidbody の値を保持して落下などの挙動を維持
        rigidbody.linearVelocity = transform.rotation * new Vector3(
            targetVelocity.x,
            rigidbody.linearVelocity.y,
            targetVelocity.y
        );
    }
}
