using UnityEngine;

public class UIManager : MonoBehaviour
{
    public GameObject canvas;

    public void HideCanvas()
    {

        // Canvas を非表示にする
        if (canvas != null)
        {
            canvas.SetActive(false);
        }
 }
}
