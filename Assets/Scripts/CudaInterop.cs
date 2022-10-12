using UnityEngine;
using System;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;

public class CudaInterop : MonoBehaviour
{
    //Call C++
    [DllImport("Cuda_Interop", EntryPoint = "SendTextureIDToCuda")]
    private static extern void SendTextureIDToCuda(int texture_id, int width, int height);
    [DllImport("Cuda_Interop", EntryPoint = "Dispose")]
    private static extern void Dispose();
    [DllImport("Cuda_Interop")]
    private static extern IntPtr GetRenderEventFunc();

    private RenderTexture rt;
    [SerializeField]
    private Texture2D colorTexture;
    [SerializeField]
    private Texture2D depthTexture;
    private RenderTexture depth_rt;
    [SerializeField]
    private Material depth_mat;
    private Camera _camera;


    // Start is called before the first frame update
    private void Start()
    {
        _camera = GetComponent<Camera>();
        _camera.depthTextureMode = _camera.depthTextureMode | DepthTextureMode.Depth;
        rt = _camera.targetTexture;
        depth_rt = new RenderTexture(rt.width, rt.height, 32);
        depthTexture = new Texture2D(rt.width, rt.height);
        colorTexture = new Texture2D(rt.width, rt.height);
    }

    private void OnEnable() 
    {
        RenderPipelineManager.endCameraRendering += PostRender;
    }

    private void OnDisable() 
    {
        RenderPipelineManager.endCameraRendering -= PostRender;
        //Dispose();
    }

    private void PostRender(ScriptableRenderContext context, Camera camera)
    {
        OnPostRender();
    }

    private void OnPostRender() 
    {
        
        //Debug.Log(colorTexture.GetNativeTexturePtr());
        RenderTexture.active = rt;
        colorTexture.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        colorTexture.Apply();
        Graphics.Blit(rt, depth_rt, depth_mat);
        depthTexture.ReadPixels(new Rect(0, 0, depth_rt.width, depth_rt.height), 0, 0);
        depthTexture.Apply();
        //Debug.Log(depthTexture.format);
        SendTextureIDToCuda((int)colorTexture.GetNativeTexturePtr(), colorTexture.width, colorTexture.height);
        GL.IssuePluginEvent(GetRenderEventFunc(), 1);
    }
}
