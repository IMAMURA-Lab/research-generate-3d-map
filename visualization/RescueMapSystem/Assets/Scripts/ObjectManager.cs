using UnityEngine;
using UnityEditor;
using System;
using System.Runtime.InteropServices;

public class ObjectManager : MonoBehaviour
{

    public TextLoader loader;
    public AttachComponent attachCom;
    public NavMeshBuilder navMeshBuild;

    public Transform navMeshRoot;

    // オブジェクト
    public GameObject mapObj;
    public GameObject rescuerObj;
    public GameObject personObj;
    public GameObject dangerAreaObj;
    public GameObject dangerAreaObstacleObj;

    // インスタンス
    public GameObject mapInstance;
    public GameObject rescuerInstance;
    public GameObject personInstance;
    public GameObject dangerAreaInstance;
    public GameObject dangerAreaObstacleInstance;

    // オブジェクトID
    public int mapId = 0;
    public int rescuerId = 1;
    public int personId = 2;
    public int dangerAreaId = 3;

    // オブジェクトの名前
    public string mapName = "Map";
    public string rescuerName = "Rescuer";
    public string personName = "Person";
    public string dangerAreaName = "DangerArea";
    public string dangerAreaObstacleName = "DangerAreaObstacle";

    // オブジェクトの座標
    public Vector3 mapPos = Vector3.zero;
    public Vector3 rescuerPos;
    public Vector3 personPos;
    public Vector3 dangerAreaPos;

    // オブジェクトの種類
    public string mapType = "original";
    public string rescuerType = "original";
    public string personType = "primitive";
    public string dangerAreaType = "original";
    public string dangerAreaObstacleType = "empty";

    // オブジェクトのパス
    public string mapPath = "Assets/Meshs/mesh_sample_take_by_tuka.obj";
    public string rescuerPath = "Assets/Mini First Person Controller/First Person Controller Minimal.prefab";
    public string personPath = "";
    public string dangerAreaPath = "Assets/Vefects/Free Fire VFX URP/Particles/VFX_Fire_01_Big_Simple.prefab";

    // Primitiveの種類
    public string personPrimitiveType = "Cube";
    public string dangerAreaPrimitiveType = "Cube";

    // オブジェクトのMaterialパス
    public string mapMaterialPath = "Assets/Meshs/mesh_sample_take_by_tuka_material0000_map_Kd.png";
    public string rescuerMaterialPath = "";
    public string personMaterialPath = "";
    public string dangerAreaMaterialPath = "";

    // オブジェクトのタグ
    public string rescuerTag = "Rescuer";
    public string personTag = "Person";

    // オブジェクトのレイヤー
    public string mapLayer = "Map";
    public string dangerAreaLayer = "Danger";
    public string dangerAreaObstacleLayer = "Danger";

    // オブジェクトを生成
    public void ObjectGenerator()
    {
        object[][] dataArray = loader.LoadData();

        foreach (var row in dataArray)
        {
            string name = (string)row[0];
            float x = (float)row[1];
            float y = (float)row[2];
            float z = (float)row[3];

            Debug.Log($"Name: {name}, Position: ({x}, {y}, {z})");
        }

        // -------------------------------
        // オブジェクトのデータ構造を初期化
        // -------------------------------
        // Map
        BaseObjectData mapObjBaseData = new BaseObjectData
        {
            id = mapId,
            name = mapName,
            position = mapPos,
            type = mapType
        };
        ObjectDataInitializer mapObjData = new ObjectDataInitializer(
            baseData: mapObjBaseData,
            path: mapPath,
            materialPath: mapMaterialPath,
            layer: mapLayer
            );
        // Rescuer
        BaseObjectData rescuerObjBaseData = new BaseObjectData
        {
            id = rescuerId,
            name = rescuerName,
            position = rescuerPos,
            type = rescuerType
        };
        ObjectDataInitializer rescuerObjData = new ObjectDataInitializer(
            baseData: rescuerObjBaseData,
            path: rescuerPath,
            tag: rescuerTag
            );
        // Person
        BaseObjectData personObjBaseData = new BaseObjectData
        {
            id = personId,
            name = personName,
            position = personPos,
            type = personType
        };
        ObjectDataInitializer personObjData = new ObjectDataInitializer(
            baseData: personObjBaseData,
            tag: personTag
            );
        // DangerArea
        BaseObjectData dangerAreaObjBaseData = new BaseObjectData
        {
            id = dangerAreaId,
            name = dangerAreaName,
            position = dangerAreaPos,
            type = dangerAreaType
        };
        ObjectDataInitializer dangerAreaObjData = new ObjectDataInitializer(
            baseData: dangerAreaObjBaseData,
            path: dangerAreaPath,
            layer: dangerAreaLayer
            );

        // ------------------------
        // オブジェクトの種類を設定
        // ------------------------
        // Map
        mapObj = SetObjType(mapName, mapType, null, mapPath);
        // Rescuer
        rescuerObj = SetObjType(rescuerName, rescuerType, null, rescuerPath);
        // Person
        personObj = SetObjType(personName, personType, personPrimitiveType, null);
        // DangerArea
        dangerAreaObj = SetObjType(dangerAreaName, dangerAreaType, null, dangerAreaPath);
        // DangerAreaObstacle
        dangerAreaObstacleObj = SetObjType(dangerAreaObstacleName, dangerAreaObstacleType, null, null);

        // -------------------------------------------------
        // オブジェクトをシーン内に配置し、パラメータを設定する
        // -------------------------------------------------
        // Map
        mapInstance = PlaceSceneAndSetUpParam(mapObj, null, mapName, mapPos, null, mapLayer);
        // Rescuer
        rescuerInstance = PlaceSceneAndSetUpParam(rescuerObj, null, rescuerName, rescuerPos, rescuerTag, null);
        // Person
        personInstance = PlaceSceneAndSetUpParam(personObj, null, personName, personPos, personTag, null);
        // DangerArea
        dangerAreaInstance = PlaceSceneAndSetUpParam(dangerAreaObj, null, dangerAreaName, dangerAreaPos, null, dangerAreaLayer);
        // DangerAreaObstacle
        dangerAreaObstacleInstance = PlaceSceneAndSetUpParam(dangerAreaObstacleObj, null, dangerAreaObstacleName, dangerAreaPos, null, dangerAreaLayer);

        // ----------------
        // Componentを付与
        // ----------------
        // Map
        // Mapオブジェクトの子オブジェクトを配列に格納
        Transform[] mapObjChildren = new Transform[mapInstance.transform.childCount];
        for (int i = 0; i < mapInstance.transform.childCount; i++)
        {
            mapObjChildren[i] = mapInstance.transform.GetChild(i);
        }
        // MeshColliderを付与
        attachCom.AttachMeshCollider(mapInstance, mapObjChildren);

        // ----------------
        // Materialを付与
        // ----------------
        // Map
        AttachMaterial(mapInstance, mapObjChildren, mapMaterialPath);

        // ----------------------------
        // オブジェクトのパラメータ調整
        // ----------------------------
        foreach (var row in dataArray)
        {
            string name = (string)row[0];
            float x = -(float)row[1];
            float y = (float)row[2];
            float z = -(float)row[3];

            Vector3 scaledPos = new Vector3(x, y, z) * 4f;

            if(name == "Rescuer")
            {
                rescuerInstance.transform.position = scaledPos;
            }
            if(name == "DangerArea")
            {
                dangerAreaInstance.transform.position = scaledPos;
            }
            if(name == "Person")
            {
                personInstance.transform.position = scaledPos;
            }
            else Debug.Log("Failed to position.");
        }

        mapInstance.transform.localScale = new Vector3(4f, 4f, 4f);
        // personInstance.transform.localScale = new Vector3(2f, 2f, 2f);
        // dangerAreaInstance.transform.localScale = new Vector3(2f, 2f, 2f);

        // y座標の調整
        float adjustY;
        adjustY = AdjustY(mapInstance, mapObjChildren, rescuerInstance, personInstance, dangerAreaInstance);
        // Capsule Collider の設定
        // 子オブジェクトにアタッチされている場合は GetComponentInChildren を使う
        CapsuleCollider capsule = rescuerInstance.GetComponentInChildren<CapsuleCollider>();
        if (capsule != null)
        {
            capsule.center = new Vector3(0f, 0.855f, 0f);   // 中心位置
            capsule.radius = 0.3f;                   // 半径
            capsule.height = 1.71f;                     // 高さ
            capsule.direction = 1;                   // 0 = X, 1 = Y, 2 = Z
        }
        else
        {
            Debug.LogWarning("[WARNING] CapsuleCollider not found on Rescuer");
        }
        navMeshBuild.SetupNavMeshObstacle(
            dangerAreaObstacleInstance,
            dangerAreaObstacleLayer,
            dangerAreaPos,
            "Box"
            );

    }

    // PrimitiveTypeの中からオブジェクトの種類を設定
    public GameObject SetObjType(string objName, string objType, string primitiveType, string objPath)
    {
        // オブジェクトの種類がprimitiveの場合
        if (objType == "primitive"){
            if (Enum.TryParse(primitiveType, out PrimitiveType type))
            {
                GameObject obj = GameObject.CreatePrimitive(type);
                return obj;
            }
            else
            {
                Debug.LogError("[ERROR] Not Found : " + primitiveType);
                return null;
            }
        }
        // オブジェクトの種類がoriginalの場合
        else if (objType == "original")
        {
            GameObject obj = AssetDatabase.LoadAssetAtPath<GameObject>(objPath);
            if (obj == null)
            {
                Debug.LogError("[ERROR] Not Fonud: " + objPath);
                return null;
            }

            return obj;
        }
        else return null;
    }

    // オブジェクトをシーンに配置し、パラメータを設定
    public GameObject PlaceSceneAndSetUpParam(
        GameObject obj,
        Transform[] objChildren, 
        string objName, 
        Vector3 spawnPos, 
        string tagName, 
        string layerName
        )
    {
        // シーン内にオブジェクトを配置
        GameObject instance;
        if (obj != null) instance = Instantiate(obj);
        else instance = new GameObject(objName);

        // オブジェクトの名前を設定
        instance.name = objName;

        // オブジェクトの座標を設定
        instance.transform.position = spawnPos;

        // MapオブジェクトならnavMeshRootの子にする
        if (objName == mapName && navMeshRoot != null)
        {
            instance.transform.SetParent(navMeshRoot, worldPositionStays: true);
        }        
        // オブジェクトのタグを設定
        if(!string.IsNullOrEmpty(tagName)) instance.tag = tagName;
        
        // オブジェクトのレイヤーを設定
        if (layerName != null)
        {
            int layer = LayerMask.NameToLayer(layerName);
            if (layer != -1)
            {
                SetLayerRecursively(instance, layer);
            }
            else
            {
                Debug.LogWarning("[WARNING] Not found: " + layerName);
            }
        }

        Debug.Log("[NOTICE] Spawn Object: " + instance.name);

        return instance;
    }

    // 指定したオブジェクトとその全ての子オブジェクトに対してレイヤーを設定する
    public void SetLayerRecursively(GameObject obj, int layer)
    {
        if (obj == null) return;

        obj.layer = layer;

        foreach (Transform child in obj.transform)
        {
            SetLayerRecursively(child.gameObject, layer);
        }
    }

    // Materialを付与
    public void AttachMaterial(GameObject instance, Transform[] objChildren, string materialPath)
    {
        Material mat = null;

        // Materialがmatファイルの時
        if (materialPath.EndsWith(".mat"))
        {
            mat = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (mat == null)
            {
                Debug.LogError("[ERROR] Not Fonud: " + materialPath);
                return;
            }
        }
        // Materialがpngファイルの時
        else if (materialPath.EndsWith(".png"))
        {
            Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(materialPath);
            if (tex == null)
            {
                Debug.LogError("[ERROR] Not Fonud: " + materialPath);
                return;
            }

            mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            mat.mainTexture = tex;
        }

        if (objChildren.Length != 0)
        {
            MeshRenderer[] childRenderers = new MeshRenderer[objChildren.Length];
            for (int i = 0; i < objChildren.Length; i++)
            {
                childRenderers[i] = objChildren[i].GetComponent<MeshRenderer>();
            }

            // null チェックして処理
            foreach (var mr in childRenderers)
            {
                if (mr != null)
                {
                    mr.material = mat; // 子だけ操作
                    Debug.Log("[NOTICE] Attach Material Child: " + instance.name);
                }
            }
        }
        else
        {
            MeshRenderer mr = instance.GetComponentInChildren<MeshRenderer>();
            if (mr != null)
            {
                mr.material = mat;
                Debug.Log("[NOTICE] Attach Material: " + instance.name);
            }
        }
    }

    // Mapオブジェクトのy座標（床の位置）調整
    public static float AdjustY(
        GameObject mapInstance,
        Transform[] mapObjChildren,
        GameObject rescuerInstance,
        GameObject personInstance,
        GameObject dangerAreaInstance
        )
    {
        float minY = float.MaxValue;

        foreach (Transform child in mapObjChildren)
        {
            GameObject objChild = child.gameObject;

            Renderer renderer = objChild.GetComponent<Renderer>();
            if (renderer == null) continue;

            float childMinY = renderer.bounds.min.y;
            if (childMinY < minY)
            {
                minY = childMinY;
            }
        }

        if (minY < 0f)
        {
            // Map
            Vector3 pos1 = mapInstance.transform.position;
            pos1.y -= minY;
            mapInstance.transform.position = pos1;
            // Rescuer
            Vector3 pos2 = rescuerInstance.transform.position;
            pos2.y -= minY;
            rescuerInstance.transform.position = pos2;
            // Person
            Vector3 pos3 = personInstance.transform.position;
            pos3.y -= minY;
            personInstance.transform.position = pos3;
            // DangerArea
            Vector3 pos4 = dangerAreaInstance.transform.position;
            pos4.y -= minY;
            dangerAreaInstance.transform.position = pos4;
            Debug.Log("[NOTICE] Adjust Y: +" + Mathf.Abs(minY));
        }
        if (minY > 0f)
        {
            // Map
            Vector3 pos1 = mapInstance.transform.position;
            pos1.y += minY;
            mapInstance.transform.position = pos1;
            // Rescuer
            Vector3 pos2 = rescuerInstance.transform.position;
            pos2.y += minY;
            rescuerInstance.transform.position = pos2;
            // Person
            Vector3 pos3 = personInstance.transform.position;
            pos3.y += minY;
            personInstance.transform.position = pos3;
            // DangerArea
            Vector3 pos4 = dangerAreaInstance.transform.position;
            pos4.y += minY;
            dangerAreaInstance.transform.position = pos4;
            Debug.Log("[NOTICE] Adjust Y: -" + Mathf.Abs(minY));
        }

        return minY;

        // // --- ここから追加 ---
        // // Ground レイヤーの Plane を探して Y を pos.y に合わせる
        // int groundLayer = LayerMask.NameToLayer("Ground");
        // GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
        // foreach (GameObject obj in allObjects)
        // {
        //     if (obj.layer == groundLayer)
        //     {
        //         Vector3 planePos = obj.transform.position;
        //         planePos.y = pos.y;  // Plane の Y を調整
        //         obj.transform.position = planePos;
        //         Debug.Log("[NOTICE] Ground Plane Y adjusted to: " + pos.y);
        //     }
        // }
        // // --- ここまで追加 ---
    }

}

