using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;

public class CudaInterop : MonoBehaviour
{
    [DllImport("Cuda_Interop.dll", EntryPoint = "SendTextureToCuda")]
    static extern void SendTextureToCuda(int texture_id, int width, int height);
    [DllImport("Cuda_Interop.dll", EntryPoint = "Dispose")]
    static extern void Dispose();

    [SerializeField]
    private int send;
    private RenderTexture rt;
    [SerializeField]
    private Texture2D colorTexture;
    private Camera _camera;


    // Start is called before the first frame update
    private void Start()
    {
        _camera = GetComponent<Camera>();
        rt = _camera.targetTexture;
        colorTexture = new Texture2D(rt.width, rt.height);
    }
    private void OnEnable() 
    {
        RenderPipelineManager.endCameraRendering += PostRender;
    }

    private void OnDisable() 
    {
        RenderPipelineManager.endCameraRendering -= PostRender;
        Dispose();
    }

    private void PostRender(ScriptableRenderContext context, Camera camera)
    {
        OnPostRender();
    }

    private void OnPostRender() 
    {
        Debug.Log(colorTexture.GetNativeTexturePtr());
        RenderTexture.active = rt;
        colorTexture.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        colorTexture.Apply();
        SendTextureToCuda((int)colorTexture.GetNativeTexturePtr(), colorTexture.width, colorTexture.height);
        //Debug.Log(colorTexture.GetNativeTexturePtr());
    }
}
