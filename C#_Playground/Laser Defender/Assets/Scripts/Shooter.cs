using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Shooter : MonoBehaviour
{
    [Header("General")]
    [SerializeField] GameObject projectilePrefab;
    [SerializeField] float projectileSpeed = 20f;
    [SerializeField] float projectileLifeTime = 5f;
    [SerializeField] float baseFireRate = 0.1f;

    [Header("AI")]
    [SerializeField] bool useAI = true;
    [SerializeField] float AIRandomFireRate = 1f;

    [HideInInspector] public bool isFiring;
    Coroutine firingCoroutine;
    AudioPlayer audioPlayer;

    void Awake() {
        audioPlayer = FindObjectOfType<AudioPlayer>();
    }
    void Start() {
        if(useAI) {isFiring = true;}
    }

    // Update is called once per frame
    void Update() {
        Fire();
    }

    void Fire() {
        if (isFiring && firingCoroutine == null) {
            firingCoroutine = StartCoroutine(FireContinuously());
        } else if (!isFiring && firingCoroutine != null) {
            StopCoroutine(firingCoroutine);
            firingCoroutine = null;
        }
    }

    IEnumerator FireContinuously() {
        while (true) {
            GameObject instance = Instantiate(projectilePrefab, transform.position, Quaternion.identity);
            Rigidbody2D instanceBody = instance.GetComponent<Rigidbody2D>();
            if (instanceBody) {
                instanceBody.velocity = transform.up * projectileSpeed;
                audioPlayer.PlayShootingClip();
            }
            Destroy(instance, projectileLifeTime);
            if (useAI) {
                yield return new WaitForSecondsRealtime(baseFireRate + Random.Range(0, AIRandomFireRate));
            } else {
                yield return new WaitForSecondsRealtime(baseFireRate);
            }
        }
    }
}
