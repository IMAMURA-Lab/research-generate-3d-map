using UnityEngine;
using UnityEngine.AI;
using Unity.AI.Navigation;

public class NavMeshBuilder : MonoBehaviour
{

    public string groundLayerName = "Ground";
    public string mapLayerName = "Map";
    public string dangerLayerName = "Danger";

    private NavMeshSurface surface;

    void Awake()
    {
        surface = GetComponent<NavMeshSurface>();
        if(surface == null)
        {
            surface = gameObject.AddComponent<NavMeshSurface>();
            Debug.Log("[NavMeshBuilder] NavMeshSurface を自動追加しました。");     
        }
        ConfigureSurface();
    }

    void ConfigureSurface()
    {
        // 1. Agent Type を Humanoid に
        // var agentSettings = NavMesh.GetSettingsByName("Humanoid");
        // if (agentSettings >= 0)
        // surface.agentTypeID = GetAgentTypeIDByName("Humanoid");

        // 2. Collect Objects: Children
        surface.collectObjects = CollectObjects.Children;
        
        // 3. 使用する Collider をベースに
        surface.useGeometry = NavMeshCollectGeometry.PhysicsColliders;

        // 4. Include Layers（Ground, Map, Danger）
        int groundLayer = LayerMask.NameToLayer(groundLayerName);
        int mapLayer = LayerMask.NameToLayer(mapLayerName);
        int dangerLayer = LayerMask.NameToLayer(dangerLayerName);
        // レイヤーをビットマスクとして合成
        surface.layerMask = (1 << groundLayer)
                          | (1 << mapLayer)
                          | (1 << dangerLayer);        
        Debug.Log("[NavMeshSurface] Configured: CollectObjects.Children + Include Layers " +
                  groundLayerName + ", " + mapLayerName + ", " + dangerLayerName);        

        // 5. Tile / Voxel override OFF
        surface.overrideTileSize = false;
        surface.overrideVoxelSize = false;
    }

    public void SetupNavMeshObstacle(GameObject instance, string layerName, Vector3 center, string shapeType)
    {
        // "Danger" レイヤーの場合のみ処理
        if (layerName != "Danger") return;

        if (instance == null)
        {
            Debug.LogWarning("[NavMeshBuilder] Instance is null. Skipping Obstacle setup.");
            return;
        }

        // 既に NavMeshObstacle が付いていれば取得、なければ追加
        NavMeshObstacle obstacle = instance.GetComponent<NavMeshObstacle>();
        if (obstacle == null)
        {
            obstacle = instance.AddComponent<NavMeshObstacle>();
        }

        // Carve を有効化（経路計算時に避ける領域として扱う）
        obstacle.carving = true;

        // Shape 設定
        switch (shapeType)
        {
            case "Box":
                obstacle.shape = NavMeshObstacleShape.Box;
                break;
            case "Capsule":
                obstacle.shape = NavMeshObstacleShape.Capsule;
                break;
            default:
                Debug.LogWarning("[NavMeshBuilder] Unknown shapeType: " + shapeType + ". Defaulting to Box.");
                obstacle.shape = NavMeshObstacleShape.Box;
                break;
        }

        // Center 設定（ローカル座標）
        obstacle.center = center;

        Debug.Log("[NavMeshBuilder] NavMeshObstacle set on " + instance.name + " | Layer: " + layerName + " | Shape: " + shapeType);
    }

    // Bakeを実行
    public void Build()
    {
        surface.BuildNavMesh();
        Debug.Log("[NavMeshSurface] Build completed");
    }
}
