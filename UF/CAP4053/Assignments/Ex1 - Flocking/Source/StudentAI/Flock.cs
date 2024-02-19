using System;
using System.Collections.Generic;
using System.Diagnostics;
using AI.SteeringBehaviors.Core;

namespace AI.SteeringBehaviors.StudentAI
{
    public class Flock
    {
        public float AlignmentStrength { get; set; }
        public float CohesionStrength { get; set; }
        public float SeparationStrength { get; set; }
        public List<MovingObject> Boids { get; protected set; }
        public Vector3 AveragePosition { get; set; }
        protected Vector3 AverageForward { get; set; }
        public float FlockRadius { get; set; }

        #region TODO
        public Flock()
        {
            AlignmentStrength = 1.0f;
            CohesionStrength = 1.0f;
            SeparationStrength = 1.0f;
            Boids = new List<MovingObject>();
            AveragePosition = Vector3.Empty;
            AverageForward = Vector3.Empty;
            FlockRadius = 20.0f;
        }

        public virtual void Update(float deltaTime)
        {
            // Calculate whole flock data
            AveragePosition = CalcAveragePosition();
            AverageForward = CalcAverageForward();

            // Update boids
            foreach (MovingObject boid in Boids)
            {
                boid.Velocity += deltaTime * boid.MaxSpeed * (CalcAlignment(boid) + CalcCohesion(boid) + CalcSeperation(boid));
                if (boid.Velocity.Length > boid.MaxSpeed) { boid.Velocity.Normalize(); }
                boid.Update(deltaTime);
            }
        }
        #endregion
        
        private Vector3 CalcAveragePosition() {
            Vector3 newAvg = Vector3.Empty;
            foreach (MovingObject boid in Boids) {
                newAvg += boid.Position;
            }
            newAvg /= Boids.Count;
            return newAvg;
        }

        private Vector3 CalcAverageForward() {
            Vector3 newForward = Vector3.Empty;
            foreach (MovingObject boid in Boids) {
                newForward += boid.Velocity;
            }
            newForward /= Boids.Count;
            return newForward;
        }

        private Vector3 CalcAlignment(MovingObject boid) {
            Vector3 alignAccel = AverageForward / boid.MaxSpeed;
            if (alignAccel.Length > 1.0f) { alignAccel.Normalize();  }
            return AlignmentStrength * alignAccel;
        }

        private Vector3 CalcCohesion(MovingObject boid) {
            Vector3 cohesionAccel = AveragePosition - boid.Position;
            float distance = cohesionAccel.Length;
            cohesionAccel.Normalize();
            if (distance < FlockRadius) {
                cohesionAccel *= distance / FlockRadius;
            }
            return CohesionStrength * cohesionAccel;
        }

        private Vector3 CalcSeperation(MovingObject boid) {
            Vector3 sum = Vector3.Empty;

            // Loop through other boids
            Vector3 vector;
            float distance;
            float safeDistance;
            foreach (MovingObject other in Boids) {
                if (other ==  boid) continue;
                vector = boid.Position - other.Position;
                distance = vector.Length;
                safeDistance = boid.SafeRadius + other.SafeRadius;

                // If possible collision
                if (distance < safeDistance) {
                    vector.Normalize();
                    vector *= (safeDistance - distance) / safeDistance;
                    sum += vector;
                }
            }

            // Normalize as required
            if (sum.Length > 1.0f) { sum.Normalize(); }

            return SeparationStrength * sum;
        }
    }
}
