using UnityEngine;
using System;
using System.Collections.Generic;

// 1行分のデータを表すクラス
[Serializable]
public class PosData
{
    public string name;
    public float x;
    public float y;
    public float z;

    public PosData(string name, float x, float y, float z)
    {
        this.name = name;
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public override string ToString()
    {
        return $"{name} {x} {y} {z}";
    }
}

public class TextLoader : MonoBehaviour
{
    public string fileName = "PositionDatas/position_data_sample_take_by_tuka"; // Resources/data_sample.txt
    private List<PosData> dataList = new List<PosData>();

    // データをロードして二次元配列として返す
    public object[][] LoadData()
    {
        dataList.Clear(); // 以前のデータをクリア

        // Resourcesから読み込む
        TextAsset textAsset = Resources.Load<TextAsset>(fileName);
        if (textAsset == null)
        {
            Debug.LogError("[ERROR] Not Found: Resources/" + fileName + ".txt");
            return new object[0][]; // 空の配列を返す
        }

        // 改行で分割
        string[] lines = textAsset.text.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string line in lines)
        {
            string[] tokens = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length != 4)
            {
                Debug.LogWarning("[ERROR] The line format is incorrect: " + line);
                continue;
            }

            string name = tokens[0];
            if (float.TryParse(tokens[1], out float x) &&
                float.TryParse(tokens[2], out float y) &&
                float.TryParse(tokens[3], out float z))
            {
                PosData data = new PosData(name, x, y, z);
                dataList.Add(data);
            }
            else
            {
                Debug.LogWarning("[ERROR] Failed to parse numbers: " + line);
            }
        }

        // List -> 二次元配列に変換
        object[][] result = new object[dataList.Count][];
        for (int i = 0; i < dataList.Count; i++)
        {
            result[i] = new object[4] { dataList[i].name, dataList[i].x, dataList[i].y, dataList[i].z };
        }

        Debug.Log("[NOTICE] 全データ数: " + dataList.Count);
        return result;
    }
}
