using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraShake : MonoBehaviour
{
    [SerializeField] float shakeDuration = 0.5f;
    [SerializeField] float shakeMagnitude = 0.25f;
    Vector3 initPos;
    void Start() {
        initPos = transform.position;
    }

    public void Play() {
        StartCoroutine(Shake());
    }

    IEnumerator Shake() {
        float elapsedTime = 0f;
        while (elapsedTime < shakeDuration) {
            transform.position = initPos + (Vector3)Random.insideUnitCircle * shakeMagnitude;
            elapsedTime += Time.deltaTime;
            yield return new WaitForEndOfFrame();
        }
        transform.position = initPos;
    }
}
