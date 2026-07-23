using UnityEngine;
using UnityEditor;
// using System;
// using System.Numerics;

// オブジェクトの基本データ構造
public struct BaseObjectData
{
    public int id; // オブジェクトID
    public string name; // オブジェクトの名前
    public Vector3 position; // オブジェクトの座標
    public string type; // オブジェクトの種類
}

// オブジェクトのデータ構造を初期化するクラス
public class ObjectDataInitializer
{
    public BaseObjectData baseData; // 共通部分
    public string path; // オブジェクトのResourcesパス
    public string materialPath; // オブジェクトのResources/Materialsパス
    public string tag; // オブジェクトのタグ
    public string layer; // オブジェクトのレイヤー

    // コンストラクタで初期化（path, tag, layer は省略可能）
    public ObjectDataInitializer(
        BaseObjectData baseData, 
        string path = null,
        string materialPath = null,
        string tag = null, 
        string layer = null
        )
    {
        this.baseData = baseData;
        this.path = path;
        this.materialPath = materialPath;
        this.tag = tag;
        this.layer = layer;
    }
}