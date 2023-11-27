using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AudioPlayer : MonoBehaviour
{
    [Header("Shooting")]
    [SerializeField] AudioClip shootingSFX;
    [SerializeField] [Range(0f, 1f)] float shootingVolume = 1f;

    [Header("Damage")]
    [SerializeField] AudioClip damageSFX;
    [SerializeField] [Range(0f, 1f)] float damageVolume = 1f;

    static AudioPlayer instance;

    void Awake() {
        ManageSingleton();
    }

    void ManageSingleton() {
        if (instance) {
            gameObject.SetActive(false);
            Destroy(gameObject);
        } else {
            instance = this;
            DontDestroyOnLoad(gameObject);
        }
    }

    public void PlayShootingClip() {
        PlayClip(shootingSFX, shootingVolume);
    }

    public void PlayDamageClip() {
        PlayClip(damageSFX, damageVolume);
    }

    void PlayClip(AudioClip clip, float volume) {
        if (clip) {
            AudioSource.PlayClipAtPoint(clip, Camera.main.transform.position, volume);
        }
    }
}
