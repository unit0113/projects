using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpriteScroller : MonoBehaviour
{
    [SerializeField] Vector2 scrollSpeed;
    Material material;

    void Awake() {
        material = GetComponent<SpriteRenderer>().material;
    }

    void Update() {
        material.mainTextureOffset += scrollSpeed * Time.deltaTime;
    }
}
