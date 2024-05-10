// German Dominguez: Robot implementation.
// Team Name: The Doombas
// Robot Name: LordOfHeresy
// CAP4053 Spring 2024

// Latest as of 04/20/2024

using System;
using System.Drawing;
using System.Collections.Generic;
using Robocode;
using Robocode.Util;
using System.Numerics;
using static CAP4053.Student.Harvester;


// This implementation used the following source as a main reference for building a FSM implementation with Robocode:
// https://github.com/syuChen1/RoboCode/blob/master/StudenRobot/ManhattanProject.cs
// 
// LordOfHeresy Bot was simplified and adapted from ManhattanProject is several ways. 
// The LowEnergy and HighEnergy states were removed as I found them unecessary in testing. Instead I prefer a "EarlyGame" and "EndGame" 
// (NOTE: Never got around to implement EarlyGame or EndGame)
// strategy based on an approximation of the total match taken by monitoring my own bot's health.
// Wall Avoidance was also buggy in ManhattanProject and not working, so I took the WallAvoider implementation and adapted it to C#:
// https://github.com/jd12/Robocode-Movement/blob/master/templateBots/WallAvoider.java
// Wall avoidance behavior can be seen during search, attack, and retreat.

namespace CAP4053.Student
{

    // Definition of the Doombot (StudentSampleBot) for Team Doombas.
    public class LordOfHeresy : TeamRobot
    {
        // Finite State Machine for the bot. Defined below.
        FSM fsm = new FSM();

        // EnemyBot instance that will be targeted by our bot. Defined below.
        EnemyBot enemyTarget;

        // Keeps track of our current direction of movement.
        private int moveDirection = 0;
        // Specifies how close we our bot can get to the walls.
        private int wallMargin = 60;
        // Specifies our robot's targeting range, will not shoot if target is further than targetDist.
        private double targetDist = 600;
        // Specifies the radius we allow any other bots to be from our bot, will move away from enemies and teammates alike.
        private double botRadius = 100;
        // Keeps track of the timing for strafing, reset after each change of direction.
        private double strafeTiming = 27; // Set to 27 and randomized bc changing direction every N ticks was too predictable.
        // Keeps track if we are too close to a wall.
        private int tooCloseToWall = 0;

        // Defines the threshold for any teammate to request backuo, when their energy becomes <= to this they can request.
        private double teamBackupEnergyThreshold = 50;

        // List of friendly bots.
        private Dictionary<string, FriendlyInfo> friends = new Dictionary<string, FriendlyInfo>();
        private double friendlyRadius = 30;


        // Random number generator
        Random rng = new Random();

        Boolean initialized = false;


        // Initialized the gun and radar as well as the default colors.
        public void InitializeBot()
        {
            // Set move direction to -1 or 1 for more randomness on game init.
            moveDirection = rng.Next(0, 2) * 2 - 1;

            IsAdjustGunForRobotTurn = true;
            IsAdjustRadarForRobotTurn = true;

            SetDefaultColors();
            initialized = true;

        }

        // Main function that runs on every update.
        public override void Run()
        {
            // Run initialize function if not initialized.
            if (!initialized)
            {
                InitializeBot();

            }
            // Add a custom event named "too_close_to_walls"
            AddCustomEvent(new Condition("too_close_to_wall", c => TooCloseToWall()));

            while (true)
            {
                // Always broadcast my friendly info.
                BroadcastMyFriendlyInfo();

                // If enemy target is null means we dont have a target yet, a.k.a still searching.
                if (enemyTarget == null)
                {
                    // Add the searching event to the FSM.
                    fsm.enqueEvent(FSM.Events.searching);
                    // Continue turning radar until we find another bot.
                    Search();

                    /**
                     * Interestingly, trying to set the state to search through a proper transition significantly reduced success w FailSharpBot.
                     * No other aspects of the code contributed to this other than this, so I call Search() manually if no bot is found.
                     * I think this has to do with the time wasted setting the searching event and going through the FSM directly instead of potentially attacking.
                     * 
                     * Essentially, the FSM doesn't actually transition to the search state, but for our purposes this is enough to get it to search and attack when necessary.
                     */
                }

                // If enemy target not null means we have a target.
                if (enemyTarget != null)
                {
                    // Add the targeting event to the FSM.
                    fsm.enqueEvent(FSM.Events.targeting);

                    // If my energy goes below 50, send a backup request to a teammate.
                    if (friends.Count > 0 && Time % 10 == 0 && Energy < 50)
                    {
                        BroadcastBackupRequest();

                    }
                }

                // Must transition on each update.
                fsm.Transition();


                // Must be called, otherwise set functions won't run.
                Execute();

                // Write the gamestate to console for debugging.
                PrintD("Game state: " + fsm.gameState);

            }
        }

        // Once a robot is scanned (teammate or enemy) will avoid collisions and run the FSM if the robot is an enemy.
        public override void OnScannedRobot(ScannedRobotEvent scannedBot)
        {

            // If the robot is not a teammate, set the bot as a targeted enemy and run the FSM.
            if (!IsTeammate(scannedBot.Name))
            {
                enemyTarget = new EnemyBot(scannedBot);

                double firePower = Math.Min((targetDist / enemyTarget.distance), 3);

                // Set FSM to ram if our energy is less than firepower, or if enemy energy is less than 45 and less than our energy.
                if (Energy < firePower || (scannedBot.Energy < 45 && scannedBot.Energy < Energy))
                {
                    // Add the canRam event to the queue.
                    fsm.enqueEvent(FSM.Events.canRam);
                }

                // Set FSM to no ram if enemy energy is greater than 45 or greater than our own energy.
                if (scannedBot.Energy > 45 || scannedBot.Energy > Energy)
                {
                    // Add the noRam event to the queue.
                    fsm.enqueEvent(FSM.Events.noRam);
                }

                // Transition the FSM
                fsm.Transition();

                // Based on the transition, allow retreat, attack, or ram.
                if (fsm.gameState == FSM.State.retreat) // @TODO make the bot retreat if we have taken too much damage recently.
                {
                    Retreat(firePower);
                }
                else if (fsm.gameState == FSM.State.attack)
                {
                    Attack(firePower);
                }
                else if (fsm.gameState == FSM.State.ram)
                {
                    Ram(firePower);
                }

            }

            // Move away from any scanned bot (enemy or teammate) if it is within our bot's radius.
            else if (scannedBot.Distance < botRadius)
            {
                moveDirection *= -1; // Change Direction.
            }

            /**
             * Interestingly, there was a noticeable drop in SUCCESS against FailBotSharp if the collision detection came before the FSM calculations for a scanned target.
             * I thought if I put the collisoon detection for any bot first it would do better and "waste less time" but it seemed to do worse, likely because it was taking longer to transition.
             * Weird.
             **/

        }

        // Function for our bot to search for a target. Should run on init and whenever we need a new target.
        public void Search()
        {
            // Should move the bot around and scan until it finds an enemy.

            // Continue turning radar until we find another bot.
            SetTurnRadarRight(double.PositiveInfinity);


            // If too close to wall is greater than 0, substract and eventually we'll move.
            if (tooCloseToWall > 0)
            {
                tooCloseToWall--;
            }
            // If not too close to wall, randomly strafe.
            else
            {
                RandomStrafe();
            }

            // switch directions if we've stopped
            // (also handles moving away from the wall if too close)
            if (Velocity == 0)
            {
                PrintD("Inside speed check should be moving away???");
                MaxVelocity = 8;
                moveDirection *= -1;
            }

            SetAhead(targetDist * moveDirection);

        }

        // Function for our bot to attack with gun.
        public void Attack(double firePower)
        {
            // http://mark.random-article.com/weber/java/robocode/lesson4.html predictive shooting @TODO

            double absbearing = Utils.NormalRelativeAngle(HeadingRadians + enemyTarget.bearingRadians);

            double bulletSpeed = 20 - firePower * 3;

            double enemyX = enemyTarget.GetX(absbearing, X); // Needs absolute bearing and my robot's X.
            double enemyY = enemyTarget.GetY(absbearing, Y); // Needs absolute bearing and my robot's Y.

            long time = (long)(enemyTarget.distance / bulletSpeed);

            double enemyFutureX = enemyTarget.GetFutureX(enemyX, time);
            double enemyFutureY = enemyTarget.GetFutureY(enemyY, time);

            double futureBearing = enemyTarget.GetFutureBearing(enemyFutureX, enemyFutureY, X, Y); // Needs both future enemy coordinates and my robot's coordinates.
            futureBearing = Utils.NormalRelativeAngle(futureBearing - HeadingRadians); // Offset this by my robot's heading in radians.

            if (futureBearing > 3.5)
            {
                futureBearing = 0;
            }
            else
            {
                futureBearing = enemyTarget.velocity * enemyTarget.GetFutureRat(HeadingRadians); // Needs heading radians to calculate.
            }

            SetTurnRadarLeftRadians(RadarTurnRemainingRadians);
            double aheadDistance = enemyTarget.distance - 130;

            // http://mark.random-article.com/weber/java/robocode/lesson5.html circle in and strafing

            double gunTurnAmount = Utils.NormalRelativeAngle(futureBearing / 16 + absbearing - GunHeadingRadians);
            SetTurnRightRadians(Utils.NormalRelativeAngle(Math.PI / 2 + enemyTarget.bearingRadians - (0.3 * moveDirection)));

            // If too close to wall is greater than 0, substract and eventually we'll move.
            if (tooCloseToWall > 0)
            {
                tooCloseToWall--;
            }
            // If not too close to the wall, randomly strafe while attacking.
            else
            {
                RandomStrafe();
            }

            // switch directions if we've stopped
            // (also handles moving away from the wall if too close)
            if (Velocity == 0)
            {
                PrintD("Inside speed check should be moving away???");
                MaxVelocity = 10;
                moveDirection *= -1;
            }

            // Set lineCastX and lineCastY
            // For some reason while debugging needed *-1 to actually point the way the gun was pointing??
            (lineCastX, lineCastY) = (enemyFutureX, enemyFutureY);

            SetTurnGunRightRadians(gunTurnAmount);
            SetAhead(aheadDistance * moveDirection);


            // Only fire if gun heat is 0, energy is greater than firepower, and the enemy distance is less than our set target distance.
            if (GunHeat == 0 && Energy > firePower && enemyTarget.distance <= targetDist)
            {
                // Only fire if no friendly fire check passes.
                if (friends.Count == 0 || NoFriendlyFire())
                {
                    SetFire(firePower);
                }

            }
        }

        // Function for our bot to ram.
        public void Ram(double firePower)
        {
            moveDirection = 1;
            double absbearing = Utils.NormalRelativeAngle(HeadingRadians + enemyTarget.bearingRadians);
            double bulletSpeed = 20 - firePower * 3;

            double enemyX = enemyTarget.GetX(absbearing, X); // Needs absolute bearing and my robot's X.
            double enemyY = enemyTarget.GetY(absbearing, Y); // Needs absolute bearing and my robot's Y.


            long time = (long)(enemyTarget.distance / bulletSpeed);

            double enemyFutureX = enemyTarget.GetFutureX(enemyX, time);
            double enemyFutureY = enemyTarget.GetFutureY(enemyY, time);

            double futureBearing = enemyTarget.GetFutureBearing(enemyFutureX, enemyFutureY, X, Y); // Needs both enemy future coordinates and my robot's coordinates.
            futureBearing = Utils.NormalRelativeAngle(futureBearing - HeadingRadians); // Offset this by my robot's heading in radians.


            if (futureBearing > 3.5)
            {
                futureBearing = 0;
            }
            else
            {
                futureBearing = enemyTarget.velocity * enemyTarget.GetFutureRat(HeadingRadians);
            }

            double gunTurnAmount;
            SetTurnRadarLeftRadians(RadarTurnRemainingRadians);
            gunTurnAmount = Utils.NormalRelativeAngle(absbearing - GunHeadingRadians + futureBearing / 16);

            SetTurnGunRightRadians(gunTurnAmount);
            SetTurnRightRadians(Utils.NormalRelativeAngle(enemyTarget.bearingRadians));

            // Set lineCastX and lineCastY
            // For some reason while debugging needed *-1 to actually point the way the gun was pointing??
            (lineCastX, lineCastY) = CalculateAttackPosition(HeadingRadians, enemyTarget.distance);

            // Only ram if no friendly fire check passes.
            if (friends.Count == 0 || NoFriendlyFire())
            {
                // Set ahead towards enemy + targetDist
                SetAhead((enemyTarget.distance + targetDist) * moveDirection);
            }
            else // If cannot ram, change direction.
            {
                moveDirection *= -1;
            }
        }

        // Function for our bot to retreat.
        public void Retreat(double firePower)
        {
            SetTurnRadarLeftRadians(RadarTurnRemainingRadians);
            SetTurnRight(Utils.NormalRelativeAngle(enemyTarget.bearingRadians + Math.PI / 2));


            // If too close to wall is greater than 0, substract and eventually we'll move.
            if (tooCloseToWall > 0)
            {
                tooCloseToWall--;
            }
            else
            {
                MaxVelocity = Math.Min((1 + rng.NextDouble()), 1.0) * 12; // @TODO

            }

            // Set the movement away from the enemy.
            SetAhead(Math.Max(100, enemyTarget.distance) * moveDirection * -1); // @TODO
        }

        // In the case where we hit a robot set tooCloseToWall to 0.
        public override void OnHitRobot(HitRobotEvent evnt)
        {
            tooCloseToWall = 0;
        }

        // In the case we hit the wall change the movement direction.
        public override void OnHitWall(HitWallEvent e)
        {
            PrintD("Oops, I hit a wall bearing " + e.Bearing + " degrees. Changing movement direction.\n");

            moveDirection *= -1;
        }

        // if enemy dies reset enemy target to null.
        public override void OnRobotDeath(RobotDeathEvent e)
        {
            PrintD("Enemy Robot dead, setting enemyTarget to null.\n");

            if (enemyTarget == null)
            {
                Search();
            }
            // Remove from the friends dic if our comrade has fallen.
            else if (friends.ContainsKey(e.Name))
            {
                friends.Remove(e.Name);
            }
            else if (e.Name == enemyTarget.name)
            {
                enemyTarget = null;

                Search();
            }

        }


        // Changes the direction (strafes) based on time ticks, sets strafeTiming to new random interval.
        private void RandomStrafe()
        {
            // While attacking, change direction every so often ticks to avoid being hit.
            if (Time % strafeTiming == 0)
            {
                moveDirection *= -1;

                strafeTiming = rng.Next(19, 37); // @TODO find sweet spot for timing
            }
        }

        // Returns true if we are too close to the walls based on the wall margin.
        private bool TooCloseToWall()
        {
            Boolean fin = false;
            if (X <= wallMargin || X >= BattleFieldWidth - wallMargin ||
                Y <= wallMargin || Y >= BattleFieldHeight - wallMargin)
            {
                fin = true;

            }

            return fin;
        }

        // Override the OnCustomEvent method to handle the custom event
        public override void OnCustomEvent(CustomEvent e)
        {
            if (e.Condition.Name == "too_close_to_wall")
            {
                // Perform actions when too close to walls
                if (tooCloseToWall <= 0)
                {
                    tooCloseToWall += wallMargin;
                    // Stop the robot
                    MaxVelocity = 0;

                }
            }
        }


        //////////////////////////////////////////
        // TEAM FUNCTIONS
        //////////////////////////////////////////

        // Helper function that runs throughall our friends coordinates, and checks they do not collide with our fire line
        public bool NoFriendlyFire()
        {
            foreach (var friend in friends)
            {
                // Each one is a struct if FriendlyInfo.
                // If there is any collision with any of our friends return false.
                FriendlyInfo friendInfo = friend.Value;
                if (CheckCollision(friendInfo.x, friendInfo.y, friendlyRadius) == true)
                {
                    PrintD("Cannot fire/ram without hurting friendly, avoid fire.");
                    return false;
                }
            }

            // Return true if all friends are safe from collision.
            PrintD("FIRE/RAM AT WILL!");
            return true;
        }

        // Helper function to broadcast MY BOT's information to other friendly
        public void BroadcastMyFriendlyInfo()
        {
            BroadcastMessage(new FriendlyInfo(Energy, X, Y, Name, Velocity, HeadingRadians));
        }

        // Helper function to broadcast the backup request to all teammates.
        public void BroadcastBackupRequest()
        {

            // Sends a backup request to attack my current targeted enemy.
            string name = enemyTarget.name;
            double energy = enemyTarget.enemyEvent.Energy;
            double bearing = enemyTarget.bearing;
            double distance = enemyTarget.distance;
            double heading = enemyTarget.enemyEvent.Heading;
            double velocity = enemyTarget.velocity;
            bool isSentry = false;

            // Send the enemy target info over a backuo request for teammates to handle.
            BroadcastMessage(new BackupRequest(name, energy, bearing, distance, heading, velocity, isSentry));
        }


        // helper function to switch enemy target to a new target, could be requested by teammate.
        public void SwitchEnemyTarget(BackupRequest newEnemyTarget)
        {
            enemyTarget = null;

            PrintD("Switching target to: " + newEnemyTarget.name);

            // Reconstruct the ScannedRobotEvent

            //  ScannedRobotEvent(string name, double energy, double bearing, double distance, double heading, double velocity, bool isSentryRobot)

            ScannedRobotEvent newScannedRobotEvent =
                new ScannedRobotEvent(newEnemyTarget.name, newEnemyTarget.energy, newEnemyTarget.bearing,
                    newEnemyTarget.distance, newEnemyTarget.heading, newEnemyTarget.velocity, newEnemyTarget.isSentry);


            enemyTarget = new EnemyBot(newScannedRobotEvent);

        }

        // Function that runs when a message is received.
        public override void OnMessageReceived(MessageEvent e)
        {
            // Must set to type for friendly info about our teammates or enemy info about other targets.
            if (e.Message.GetType() == typeof(FriendlyInfo))
            {
                FriendlyInfo info = (FriendlyInfo)e.Message;
                // Add the friendlyInfo to the friends dictionary.
                friends[info.name] = info;

                PrintD("New Friendly info received: " + e.Message);

            }
            else if (e.Message is BackupRequest)
            {
                PrintD("RequestBackup message received, parsing.");

                BackupRequest backupRequest = (BackupRequest)e.Message;
                // Access the contents of the message
                SwitchEnemyTarget(backupRequest);
            }

        }


        // Helper struct to contain the data to be sent.
        [Serializable]
        public struct BackupRequest
        {
            public string name;
            public double energy;
            public double bearing;
            public double heading;
            public double distance;
            public double velocity;
            public bool isSentry;


            public BackupRequest(string _name, double _energy, double _bearing, double _distance, double _heading, double _velocity, bool _isSentry)
            {
                name = _name;
                energy = _energy;
                bearing = _bearing;
                distance = _distance;
                heading = _heading;
                velocity = _velocity;
                isSentry = _isSentry;
            }


        }

        // Friendly Teammate Info struct, this is what each of us share to our team about ourselves.
        [Serializable]
        public struct FriendlyInfo
        {
            // potential additions
            // targeted enemy
            // gun direction
            // radar direction
            // heading
            // speed
            // future plans?

            public FriendlyInfo(double _energy, double _x, double _y, string _name, double _vel, double _heading)
            {

                energy = _energy;
                x = _x;
                y = _y;
                name = _name;
                velocity = _vel;
                heading = _heading;
            }

            public double energy { get; }
            public double x { get; }
            public double y { get; }
            public string name { get; }
            public double velocity { get; }
            public double heading { get; }
        }


        //////////////////////////////////////////
        /// Debug Functions
        //////////////////////////////////////////

        // Helper function that checks for a collision given a friendly's x,y coordinate and a radius
        public bool CheckCollision(double friendlyX, double friendlyY, double radius)
        {
            // (x1,y1) are my coordinates.
            double myX = X;
            double myY = Y;

            // (x2, y2) are the calculated gun coordinates.
            double x2 = lineCastX;
            double y2 = lineCastY; // lineCastX and lineCastY are calculated in each frame for the  attack function.

            double a, b, c;
            (a, b, c) = CalculateLineCoefficients(myX, myY, x2, y2);

            // Finding the distance of line from center.
            double dist = (Math.Abs(a * friendlyX + b * friendlyY + c)) /
                            Math.Sqrt(a * a + b * b);

            // Checking if the distance is less than, 
            // greater than or equal to radius.
            if (radius == dist)
            {
                Console.WriteLine("Touch");
                return true;
            }

            else if (radius > dist)
            {
                Console.WriteLine("Intersect");
                return true;
            }
            else
            {
                Console.WriteLine("Outside");
                return false;
            }

        }

        // Helper function that returns the a, b, c coefficients for the line of fire from (x1,y1) and (x2,y2)
        public (double a, double b, double c) CalculateLineCoefficients(double x1, double y1, double x2, double y2)
        {
            double a = y1 - y2;
            double b = x2 - x1;

            double c = (x1 - x2) * y1 + (y2 - y1) * x1;

            return (a, b, c);
        }

        // Calculates the x and y position a potential shot fired from my gun given a heading and a distance (usually the targets).
        public (double x, double y) CalculateAttackPosition(double gunHeading, double distance)
        {
            // Calculate the angle of the gun
            double absoluteGunAngle = (gunHeading + Math.PI / 2) % (2 * Math.PI);

            // Calculate the x and y coordinates of the gun position
            double x = X + distance * Math.Cos(absoluteGunAngle);
            double y = Y + distance * Math.Sin(absoluteGunAngle);

            return (x, y);
        }


        // The coordinates of the last scanned robot
        double lineCastX = -1;
        double lineCastY = -1;

        // Paint a transparent square on top of the last scanned robot
        public override void OnPaint(IGraphics graphics)
        {

            // Set the paint color to a red half transparent color
            var redBrush = new SolidBrush(Color.FromArgb(80, Color.Red));

            var redPen = new Pen(redBrush);

            // Draw a line from our robot to the scanned robot
            graphics.DrawLine(redPen, (int)lineCastX, (int)lineCastY, (int)X, (int)Y);

            // Draw a filled square on top of the scanned robot that covers it
            graphics.FillRectangle(redBrush, (int)lineCastX - 20, (int)lineCastY - 20, 40, 40);

            Color green = Color.FromArgb(80, Color.Green);
            var greenBrush = new SolidBrush(green);

            var greenPen = new Pen(greenBrush);
            // Draw a circle of my radius in green.
            graphics.DrawEllipse(greenPen, (float)(X - friendlyRadius), (float)(Y - friendlyRadius), (float)(friendlyRadius * 2), (float)(friendlyRadius * 2));
            // Fill the circle
            graphics.FillEllipse(greenBrush, (float)(X - friendlyRadius), (float)(Y - friendlyRadius), (float)(friendlyRadius * 2), (float)(friendlyRadius * 2));
        }


        // Helper function to print a message to the robots console.
        public void PrintD(String message)
        {
            Out.WriteLine(message);
        }

        // Helper to set colors.
        public void SetDefaultColors()
        {

            // Set team colors.
            Color body = (Color.FromArgb(28, 59, 51));
            Color gun = (Color.FromArgb(113, 106, 78));
            Color radar = (Color.FromArgb(180, 94, 51));
            Color scan = (Color.FromArgb(78, 59, 53));
            Color bullet = (Color.FromArgb(108, 75, 54));


            SetColors(body, gun, bullet, radar, scan);

        }

        //////////////////////////////////////////

        // Definition of our EnemyBot class.
        public class EnemyBot
        {
            internal ScannedRobotEvent enemyEvent;

            // Name, bearing, and distance of the enemy.
            internal string name;
            internal double bearing;
            internal double distance;
            internal double velocity;
            internal double bearingRadians;

            internal EnemyBot(ScannedRobotEvent e)
            {
                this.enemyEvent = e;
                this.name = e.Name;
                this.bearing = e.Bearing;
                this.distance = e.Distance;
                this.velocity = e.Velocity;
                this.bearingRadians = e.BearingRadians;
            }

            // Gets the enemies X coordinate based on the absolute bearing and my robots X coordinate.
            internal double GetX(double absBearing, double robotX)
            {
                double enemyX = 0;

                // Target on the right
                if (absBearing > 0)
                {
                    if (absBearing > Math.PI / 2)
                    {
                        enemyX = robotX + Math.Cos(absBearing - Math.PI / 2) * this.distance;
                    }

                    else if (absBearing <= Math.PI / 2)
                    {
                        enemyX = robotX + Math.Cos(Math.PI / 2 - absBearing) * this.distance;
                    }

                }

                // Target is on the left
                if (absBearing < 0)
                {
                    if (absBearing > Math.PI / 2)
                    {
                        enemyX = robotX - Math.Cos(absBearing + Math.PI / 2 * -1) * this.distance;
                    }

                    else if (absBearing <= Math.PI / 2)
                    {
                        enemyX = robotX - Math.Cos(absBearing + Math.PI / 2) * this.distance;
                    }


                }



                return enemyX;
            }

            // Gets the enemies Y coordinate based on the absolute bearing and my robots Y coordinate.
            internal double GetY(double absbearing, double robotY)
            {
                double enemyY = 0;

                // Target is below us.
                if (Math.Abs(absbearing) > Math.PI / 2)
                {
                    enemyY = robotY - Math.Cos(Math.PI - Math.Abs(absbearing)) * this.distance;
                }

                // Target is above us.
                else if (Math.Abs(absbearing) <= Math.PI / 2)
                {
                    enemyY = robotY + Math.Cos(Math.Abs(absbearing)) * this.distance;

                }


                return enemyY;
            }



            // Internal functions for the enemy bot that get's it's future X position pased on my robots X position.
            internal double GetFutureX(double enemyX, double time)
            {
                double futureX = 0;

                // Target is going right.
                if (enemyEvent.HeadingRadians > 0)
                {
                    if (enemyEvent.HeadingRadians <= Math.PI / 2)
                    {
                        futureX = enemyX + Math.Cos(Math.PI / 2 - enemyEvent.HeadingRadians) * enemyEvent.Velocity * time;
                    }

                    else if (enemyEvent.HeadingRadians > Math.PI / 2)
                    {
                        futureX = enemyX + Math.Cos(enemyEvent.HeadingRadians - Math.PI / 2) * enemyEvent.Velocity * time;
                    }
                }


                // Target is going left
                else if (enemyEvent.HeadingRadians < 0)
                {
                    if (enemyEvent.HeadingRadians <= Math.PI / 2)
                    {
                        futureX = enemyX - Math.Cos(enemyEvent.HeadingRadians + Math.PI / 2) * enemyEvent.Velocity * time;
                    }
                    else if (enemyEvent.HeadingRadians > Math.PI / 2)
                    {
                        futureX = enemyX - Math.Cos((enemyEvent.HeadingRadians + Math.PI / 2) * -1) * enemyEvent.Velocity * time;
                    }
                }

                return futureX;
            }

            // Internal functions for the enemy bot that get's it's future Y position pased on my robots Y position.
            internal double GetFutureY(double enemyY, double time)
            {
                double futureY = 0;

                // Target is going down.
                if (Math.Abs(enemyEvent.HeadingRadians) > Math.PI / 2)
                {
                    futureY = enemyY - Math.Cos(180 - Math.Abs(enemyEvent.HeadingRadians)) * enemyEvent.Velocity * time;
                }

                // Target is going up.
                else if (Math.Abs(enemyEvent.HeadingRadians) <= Math.PI / 2)
                {
                    futureY = enemyY + Math.Cos(Math.Abs(enemyEvent.HeadingRadians)) * enemyEvent.Velocity * time;

                }

                return futureY;
            }

            // Gets a future bearing of the enemy based on it's future X and Y and my robot's current position.
            internal double GetFutureBearing(double enemyFutureX, double enemyFutureY, double robotX, double robotY)
            {
                double deltaY = enemyFutureY - robotY; // Change in Y from enemy to my robot.
                double deltaX = enemyFutureX - robotX; // Change in X from enemy to my robot.

                double bearing = Math.Atan2(deltaY, deltaX); // Requires double (Y,X)

                if (deltaX >= 0 && deltaY >= 0)
                {
                    bearing = Math.PI / 2 - bearing;
                }
                else if (deltaX >= 0 && deltaY <= 0)
                {
                    bearing = Math.Abs(bearing) + Math.PI / 2;
                }
                else if (deltaX <= 0 && deltaY <= 0)
                {
                    bearing = -(bearing - Math.PI / 2);
                }
                else if (deltaX <= 0 && deltaY >= 0)
                {
                    bearing = -(bearing + Math.PI / 2 * 3);
                }

                return bearing;
            }

            // @TODO wtf does this do lmao.
            internal double GetFutureRat(double HeadingRadians)
            {
                double absheading = HeadingRadians + this.bearingRadians;

                double rat = Math.Sin(this.enemyEvent.HeadingRadians - absheading);
                return rat;
            }

        }

    }


    // Definition of our Finite State Machine
    //https://stackoverflow.com/questions/5923767/simple-state-machine-example-in-c Finite State Machine @TODO

    public class FSM
    {
        // The different states for our bot.
        public enum State
        {
            search,     // Search state that roams and looks for a target. On init and when target is eliminated and search begins again.
            attack,     // Default attack state (uses gun)
            ram,        // Ram attack state (when energy is low for enemy)
            retreat,    // State to retreat to safety when too close to an enemy.
        }

        // Events that can trigger the different search states.
        public enum Events
        {
            searching,  // Searching will be on init, for searching for targets.
            targeting,  // Targeting will be once a target is found, we start an attack pattern.
            canRam,     // Can ram event triggers when targeted enemy is able to be rammed. (Based on energy etc.)
            noRam,      // No ram triggers when targeted enemy is not able to be rammed or should not be rammed. (based on energy or late game preference)
        }

        private Action[,] transitionTable;
        public State gameState { get; set; }
        Queue<Events> queue = new Queue<Events>();

        public FSM()
        {
            // FSM needs a transition table that manages how one state leads to the next.
            this.transitionTable = new Action[4, 4]
            {   // Targeting,     // Searching     // Can Ram         // No Ram    
                {this.toAttack,   this.toSearch,   this.toRam,        this.toAttack,  }, // default search
                {this.toAttack,   this.toSearch,   this.toRam,        this.toAttack,  }, // default attack
                {this.toRam,      this.toSearch,   this.toRam,        this.toAttack,  }, // default ram
                {this.toRetreat,  this.toSearch,   this.toRetreat,    this.toRetreat, }  // default retreat
            };

            // Adding transition from search to itself
            this.transitionTable[(int)State.search, (int)Events.searching] = this.toSearch;
        }



        // Function to add event to the queue.
        public void enqueEvent(Events e)
        {
            queue.Enqueue(e);
        }

        // Transition function necessary for each update, handles switch from state to state.
        public void Transition()
        {
            // Handle transition if the queue is not empty.
            if (queue.Count > 0)
            {
                this.transitionTable[(int)this.gameState, (int)queue.Dequeue()].Invoke();
            }
        }

        // Functions to set the states for search, attack, ram, and retreat.
        private void toSearch()
        {
            this.gameState = State.search;
        }
        private void toAttack()
        {
            this.gameState = State.attack;
        }
        private void toRam()
        {
            this.gameState = State.ram;
        }
        private void toRetreat()
        {
            this.gameState = State.retreat;
        }

    }

}

