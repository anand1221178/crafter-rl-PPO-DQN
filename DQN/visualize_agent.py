"""
Visualize trained Dyna-Q agent playing Crafter.

This script loads a trained model and either:
1. Records videos of the agent playing
2. Shows real-time gameplay in a window

Usage:
    # Record videos
    python visualize_agent.py --model models/dynaq/dynaq_baseline_eval1.pt --episodes 5 --save_video

    # Watch in real-time
    python visualize_agent.py --model models/dynaq/dynaq_baseline_eval1.pt --episodes 3
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from src.agents.dynaq_agent import DynaQAgent
import imageio  # For saving videos
import crafter  # Crafter environment


def visualize_agent(model_path, episodes=5, save_video=True, video_folder="videos"):
    """
    Watch trained agent play Crafter.

    Args:
        model_path: Path to trained model (.pt file)
        episodes: Number of episodes to record/play
        save_video: If True, save videos; if False, display in window
        video_folder: Where to save videos
    """

    print(f"🎮 Loading model: {model_path}")

    # Setup environment (use crafter.Env() directly like in training)
    env = crafter.Env()

    if save_video:
        # Create video folder
        Path(video_folder).mkdir(exist_ok=True)
        print(f"📹 Recording videos to: {video_folder}/")

    # Load agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obs = env.reset()  # Old gym returns obs directly, not (obs, info)
    observation_shape = obs.shape
    num_actions = env.action_space.n

    agent = DynaQAgent(
        observation_shape=observation_shape,
        num_actions=num_actions,
        device=device
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    agent.q_network.load_state_dict(checkpoint['q_network'])
    agent.q_network.eval()

    print(f"✅ Model loaded successfully")
    print(f"📊 Playing {episodes} episodes...")
    print()

    # Action names for logging
    ACTION_NAMES = [
        'Noop', 'Move Left', 'Move Right', 'Move Up', 'Move Down',
        'Do', 'Sleep', 'Place Stone', 'Place Table', 'Place Furnace',
        'Place Plant', 'Make Wood Pickaxe', 'Make Stone Pickaxe',
        'Make Iron Pickaxe', 'Make Wood Sword', 'Make Stone Sword',
        'Make Iron Sword'
    ]

    all_episode_stats = []

    for ep in range(episodes):
        obs = env.reset()  # Old gym: returns obs directly
        done = False
        total_reward = 0
        steps = 0
        achievements = []
        action_counts = {i: 0 for i in range(num_actions)}

        # For video recording
        frames = [] if save_video else None

        print(f"{'='*60}")
        print(f"Episode {ep+1}/{episodes}")
        print(f"{'='*60}")

        while not done:
            # Save frame for video
            if save_video:
                frame = env.render()  # Get RGB frame
                if frame is not None:
                    frames.append(frame)

            # Agent acts greedily (no exploration)
            action = agent.act(obs, training=False)
            action_counts[action] += 1

            # Old gym: returns (obs, reward, done, info) not 5 values
            obs, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1

            # Track achievements
            if reward > 0:
                # Log positive rewards (achievements)
                print(f"  Step {steps}: {ACTION_NAMES[action]} → Reward: +{reward:.2f}")

                # Try to extract achievement name from info
                for key, value in info.items():
                    if key.startswith('achievement_') and value > 0:
                        achievement_name = key.replace('achievement_', '').replace('_', ' ').title()
                        achievements.append(achievement_name)

            # Periodic status updates (every 100 steps)
            if steps % 100 == 0:
                print(f"  Step {steps}: Reward={total_reward:.2f}, Still alive...")

        # Episode summary
        print(f"\n{'─'*60}")
        print(f"📊 Episode {ep+1} Summary:")
        print(f"  Total Reward:   {total_reward:.2f}")
        print(f"  Episode Length: {steps} steps")
        print(f"  Achievements:   {len(achievements)}")

        if achievements:
            print(f"  Unlocked: {', '.join(set(achievements))}")
        else:
            print(f"  Unlocked: None")

        # Show most common actions
        top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:5]
        print(f"\n  🎯 Most Used Actions:")
        for action_id, count in top_actions:
            if count > 0:
                pct = 100 * count / steps
                print(f"     {ACTION_NAMES[action_id]:<20} {count:4d} ({pct:5.1f}%)")

        print(f"{'─'*60}\n")

        # Save video if recording
        if save_video and frames:
            video_path = Path(video_folder) / f"episode_{ep+1:03d}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            print(f"  💾 Video saved: {video_path}")
            print()

        all_episode_stats.append({
            'episode': ep + 1,
            'reward': total_reward,
            'length': steps,
            'achievements': len(achievements),
            'achievement_names': list(set(achievements))
        })

    # Crafter's Env doesn't have close() method, so skip it
    # env.close()

    # Overall summary
    print(f"\n{'='*60}")
    print(f"📈 OVERALL SUMMARY ({episodes} episodes)")
    print(f"{'='*60}")

    avg_reward = np.mean([s['reward'] for s in all_episode_stats])
    avg_length = np.mean([s['length'] for s in all_episode_stats])
    avg_achievements = np.mean([s['achievements'] for s in all_episode_stats])

    print(f"Average Reward:       {avg_reward:.2f}")
    print(f"Average Length:       {avg_length:.1f} steps")
    print(f"Average Achievements: {avg_achievements:.1f}")

    # Unique achievements across all episodes
    all_achievements = set()
    for stat in all_episode_stats:
        all_achievements.update(stat['achievement_names'])

    if all_achievements:
        print(f"\nUnique Achievements Unlocked: {len(all_achievements)}")
        for achievement in sorted(all_achievements):
            print(f"  • {achievement}")
    else:
        print(f"\n⚠️  No achievements unlocked in any episode")

    if save_video:
        print(f"\n✅ Videos saved to: {video_folder}/")
        print(f"   Open the .mp4 files to watch the agent play!")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Dyna-Q agent playing Crafter")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5)"
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save videos instead of displaying in window"
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default="videos",
        help="Folder to save videos (default: videos/)"
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Error: Model not found at {args.model}")
        exit(1)

    visualize_agent(
        model_path=args.model,
        episodes=args.episodes,
        save_video=args.save_video,
        video_folder=args.video_folder
    )
