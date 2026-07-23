// FirstPersonAudio.cs
// 一人称キャラクターに関する「足音・走行音・着地音・ジャンプ音・しゃがみ音」など
// 状態に応じた効果音を再生・制御するスクリプト

using System.Linq;
using UnityEngine;

public class FirstPersonAudio : MonoBehaviour
{
    // プレイヤーの移動制御スクリプト
    public FirstPersonMovement character;

    // 接地判定用スクリプト（地面にいるかどうか）
    public GroundCheck groundCheck;

    // ===== 移動音関連 =====
    [Header("Step")]
    // 歩行音
    public AudioSource stepAudio;

    // 走行音
    public AudioSource runningAudio;

    // 音を鳴らすために必要な最小移動量
    [Tooltip("Minimum velocity for moving audio to play")]
    /// <summary> "Minimum velocity for moving audio to play" </summary>
    public float velocityThreshold = .01f;

    // 前フレームのキャラクター位置（XZ平面）
    Vector2 lastCharacterPosition;

    // 現在のキャラクター位置（XZ平面のみ）
    Vector2 CurrentCharacterPosition =>
        new Vector2(character.transform.position.x, character.transform.position.z);

    // ===== 着地音関連 =====
    [Header("Landing")]
    public AudioSource landingAudio;
    public AudioClip[] landingSFX;

    // ===== ジャンプ音関連 =====
    [Header("Jump")]
    public Jump jump;
    public AudioSource jumpAudio;
    public AudioClip[] jumpSFX;

    // ===== しゃがみ音関連 =====
    [Header("Crouch")]
    public Crouch crouch;
    public AudioSource crouchStartAudio, crouchedAudio, crouchEndAudio;
    public AudioClip[] crouchStartSFX, crouchEndSFX;

    // 移動中に使われるすべてのAudioSourceをまとめた配列
    AudioSource[] MovingAudios =>
        new AudioSource[] { stepAudio, runningAudio, crouchedAudio };

    void Reset()
    {
        // インスペクタで設定されていない場合の自動初期化処理

        // 親オブジェクトから移動スクリプトを取得
        character = GetComponentInParent<FirstPersonMovement>();

        // GroundCheckを子オブジェクトから取得
        groundCheck = (transform.parent ?? transform)
            .GetComponentInChildren<GroundCheck>();

        // 各種AudioSourceを取得 or 新規作成
        stepAudio = GetOrCreateAudioSource("Step Audio");
        runningAudio = GetOrCreateAudioSource("Running Audio");
        landingAudio = GetOrCreateAudioSource("Landing Audio");

        // ===== ジャンプ音の初期化 =====
        jump = GetComponentInParent<Jump>();
        if (jump)
        {
            jumpAudio = GetOrCreateAudioSource("Jump audio");
        }

        // ===== しゃがみ音の初期化 =====
        crouch = GetComponentInParent<Crouch>();
        if (crouch)
        {
            crouchStartAudio = GetOrCreateAudioSource("Crouch Start Audio");
            crouchStartAudio = GetOrCreateAudioSource("Crouched Audio");
            crouchStartAudio = GetOrCreateAudioSource("Crouch End Audio");
        }
    }

    void OnEnable()
    {
        // イベント登録（着地・ジャンプ・しゃがみ）
        SubscribeToEvents();
    }

    void OnDisable()
    {
        // イベント解除（メモリリーク防止）
        UnsubscribeToEvents();
    }

    void FixedUpdate()
    {
        // ===== キャラクターの移動量を計算 =====
        float velocity = Vector3.Distance(CurrentCharacterPosition, lastCharacterPosition);

        // 一定以上動いていて、なおかつ地面に接地している場合
        if (velocity >= velocityThreshold && groundCheck && groundCheck.isGrounded)
        {
            // しゃがみ中 → しゃがみ移動音
            if (crouch && crouch.IsCrouched)
            {
                SetPlayingMovingAudio(crouchedAudio);
            }
            // 走っている → 走行音
            else if (character.IsRunning)
            {
                SetPlayingMovingAudio(runningAudio);
            }
            // 通常移動 → 歩行音
            else
            {
                SetPlayingMovingAudio(stepAudio);
            }
        }
        else
        {
            // 移動していない or 空中 → 移動音を止める
            SetPlayingMovingAudio(null);
        }

        // 次フレーム用に現在位置を保存
        lastCharacterPosition = CurrentCharacterPosition;
    }

    /// <summary>
    /// 移動用AudioSourceを1つだけ再生し、他はすべて停止する
    /// </summary>
    void SetPlayingMovingAudio(AudioSource audioToPlay)
    {
        // 再生対象以外の移動音をすべてPause
        foreach (var audio in MovingAudios.Where(audio => audio != audioToPlay && audio != null))
        {
            audio.Pause();
        }

        // 再生対象があり、まだ再生されていなければ再生
        if (audioToPlay && !audioToPlay.isPlaying)
        {
            audioToPlay.Play();
        }
    }

    #region 即時再生系オーディオ
    // イベント発生時に1回だけ鳴る効果音群
    void PlayLandingAudio() => PlayRandomClip(landingAudio, landingSFX);
    void PlayJumpAudio() => PlayRandomClip(jumpAudio, jumpSFX);
    void PlayCrouchStartAudio() => PlayRandomClip(crouchStartAudio, crouchStartSFX);
    void PlayCrouchEndAudio() => PlayRandomClip(crouchEndAudio, crouchEndSFX);
    #endregion

    #region イベント登録・解除
    void SubscribeToEvents()
    {
        // 接地時に着地音
        groundCheck.Grounded += PlayLandingAudio;

        // ジャンプ時にジャンプ音
        if (jump)
        {
            jump.Jumped += PlayJumpAudio;
        }

        // しゃがみ開始・終了音
        if (crouch)
        {
            crouch.CrouchStart += PlayCrouchStartAudio;
            crouch.CrouchEnd += PlayCrouchEndAudio;
        }
    }

    void UnsubscribeToEvents()
    {
        // 接地イベント解除
        groundCheck.Grounded -= PlayLandingAudio;

        // ジャンプイベント解除
        if (jump)
        {
            jump.Jumped -= PlayJumpAudio;
        }

        // しゃがみイベント解除
        if (crouch)
        {
            crouch.CrouchStart -= PlayCrouchStartAudio;
            crouch.CrouchEnd -= PlayCrouchEndAudio;
        }
    }
    #endregion

    #region ユーティリティ関数
    /// <summary>
    /// 指定名のAudioSourceを探し、無ければ新規作成する
    /// </summary>
    AudioSource GetOrCreateAudioSource(string name)
    {
        // 既存のAudioSourceを検索
        AudioSource result = System.Array.Find(
            GetComponentsInChildren<AudioSource>(),
            a => a.name == name
        );

        if (result)
            return result;

        // 無ければ新規作成
        result = new GameObject(name).AddComponent<AudioSource>();
        result.spatialBlend = 1;      // 3Dサウンド
        result.playOnAwake = false;   // 自動再生しない
        result.transform.SetParent(transform, false);
        return result;
    }

    /// <summary>
    /// AudioClip配列からランダムに1つ再生
    /// </summary>
    static void PlayRandomClip(AudioSource audio, AudioClip[] clips)
    {
        if (!audio || clips.Length <= 0)
            return;

        // ランダムにクリップ選択
        AudioClip clip = clips[Random.Range(0, clips.Length)];

        // 同じ音が連続しないようにする
        if (clips.Length > 1)
            while (clip == audio.clip)
                clip = clips[Random.Range(0, clips.Length)];

        audio.clip = clip;
        audio.Play();
    }
    #endregion 
}
