import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self):
        # MUST exactly match the trained rl_games actor architecture
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(33, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

        self.act = nn.ELU()
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = (self.fc4(x))
        return x
    




def convert_network():
    # ------------------------------------------------------------
    # Load rl_games PPO checkpoint
    # ------------------------------------------------------------
    checkpoint = torch.load("/home/control-lab/aerial-gym-docker/aerialgym_ws_v2/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs/gen_ppo_13-21-06-58/nn/gen_ppo.pth", map_location="cpu")

    # rl_games stores weights under "model"
    model_state_dict = checkpoint["model"]

    print(model_state_dict.keys())

    # ------------------------------------------------------------
    # Map rl_games keys → our ActorNetwork keys
    # ------------------------------------------------------------
    mapped_state_dict = {
        "fc1.weight": model_state_dict["a2c_network.actor_mlp.0.weight"],
        "fc1.bias":   model_state_dict["a2c_network.actor_mlp.0.bias"],

        "fc2.weight": model_state_dict["a2c_network.actor_mlp.2.weight"],
        "fc2.bias":   model_state_dict["a2c_network.actor_mlp.2.bias"],

        "fc3.weight": model_state_dict["a2c_network.actor_mlp.4.weight"],
        "fc3.bias":   model_state_dict["a2c_network.actor_mlp.4.bias"],
        "fc4.weight": model_state_dict["a2c_network.mu.weight"],
        "fc4.bias": model_state_dict["a2c_network.mu.bias"]
    }

    # ------------------------------------------------------------
    # Initialize and load model
    # ------------------------------------------------------------
    actor_model = ActorNetwork()
    actor_model.load_state_dict(mapped_state_dict, strict=True)
    actor_model.eval()

    # ------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------
    sample_input = torch.randn(1, 33)
    with torch.no_grad():
        pytorch_output = actor_model(sample_input)

    print("PyTorch output shape:", pytorch_output.shape)

    # ------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------
    torch.onnx.export(
        actor_model,
        sample_input,
        "13-21-06-58.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actor_latent"],
        dynamic_axes={
            "obs": {0: "batch"},
            "actor_latent": {0: "batch"},
        },
    )

    print("✅ Exported ONNX model: gen_ppo_actor.onnx")


if __name__ == "__main__":
    convert_network()
