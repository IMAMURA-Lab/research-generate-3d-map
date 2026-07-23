using UnityEngine;
using UnityEngine.AI;

public class PathManager : MonoBehaviour
{
    [Header("対象オブジェクト")]
    private Transform rescuer; // 移動するキャラクター
    private Transform person;  // 目的地

    [Header("描画")]
    public LineRenderer lineRenderer; // 経路描画

    private NavMeshPath path; // NavMesh.CalculatePathで利用

    void Awake()
    {
        if (rescuer == null) rescuer = GameObject.FindGameObjectWithTag("Rescuer")?.transform;
        if (person == null) person = GameObject.FindGameObjectWithTag("Person")?.transform;
          
        path = new NavMeshPath();

        if(lineRenderer == null)
        {
            // LineRenderer がアタッチされていない場合は自動生成
            GameObject lrObj = new GameObject("PathLine");
            lrObj.transform.SetParent(transform, false);
            lineRenderer = lrObj.AddComponent<LineRenderer>();

            // 見た目の設定（例）
            lineRenderer.startWidth = 0.1f;
            lineRenderer.endWidth = 0.1f;
            lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
            lineRenderer.positionCount = 0;
            lineRenderer.startColor = Color.green;
            lineRenderer.endColor = Color.green;
        }
    }

    void Update()
    {
        if (rescuer == null || person == null) return;

        // NavMesh上の経路を計算
        NavMesh.CalculatePath(rescuer.position, person.position, NavMesh.AllAreas, path);

        float lineYOffset = 0.3f; // 表示用に y を 0.2m 上にずらす

        // LineRenderer で描画
        if(path.status == NavMeshPathStatus.PathComplete)
        {
            lineRenderer.positionCount = path.corners.Length;

            // コピーして Y をオフセット
            Vector3[] offsetCorners = new Vector3[path.corners.Length];
            for (int i = 0; i < path.corners.Length; i++)
            {
                offsetCorners[i] = path.corners[i] + Vector3.up * lineYOffset;
            }

            lineRenderer.SetPositions(offsetCorners);
        }
        else
        {
            // 経路が無効なら線を消す
            lineRenderer.positionCount = 0;
        }
    }
}
