using UnityEngine;

public class AttachComponent : MonoBehaviour
{
    public void AttachMeshFilter(GameObject obj, Transform[] objChildren)
    {
        if (objChildren.Length == 0)
        {
            if (obj.GetComponentInChildren<MeshFilter>() == null)
            {
                obj.AddComponent<MeshFilter>();
                Debug.Log("[NOTICE] Attach MeshFilter: " + obj.name);
            }
        }
        else
        {
            foreach (Transform child in objChildren)
            {
                GameObject objChild = child.gameObject;

                if (objChild.GetComponent<MeshFilter>() == null)
                {
                    objChild.AddComponent<MeshFilter>();
                    Debug.Log("[NOTICE] Attach MeshFilter Child: " + objChild.name);
                }
            }
        }
    }

    public void AttachMeshRenderer(GameObject obj, Transform[] objChildren)
    {
        if (objChildren.Length == 0)
        {
            if (obj.GetComponentInChildren<MeshRenderer>() == null)
            {
                obj.AddComponent<MeshRenderer>();
                Debug.Log("[NOTICE] Attach MeshRenderer: " + obj.name);
            }
        }
        else
        {
            foreach (Transform child in objChildren)
            {
                GameObject objChild = child.gameObject;

                if (objChild.GetComponent<MeshRenderer>() == null)
                {
                    objChild.AddComponent<MeshRenderer>();
                    Debug.Log("[NOTICE] Attach MeshRenderer Child: " + objChild.name);
                }
            }
        }
    }

    public void AttachMeshCollider(GameObject obj, Transform[] objChildren)
    {
        if (objChildren.Length == 0)
        {
            if (obj.GetComponentInChildren<MeshCollider>() == null)
            {
                obj.AddComponent<MeshCollider>();
                Debug.Log("[NOTICE] Attach MeshCollider: " + obj.name);
            }
        }
        else
        {
            foreach (Transform child in objChildren)
            {
                GameObject objChild = child.gameObject;

                if (objChild.GetComponent<MeshCollider>() == null)
                {
                    objChild.AddComponent<MeshCollider>();
                    Debug.Log("[NOTICE] Attach MeshCollider Child: " + objChild.name);
                }
            }
        }
    }
}
