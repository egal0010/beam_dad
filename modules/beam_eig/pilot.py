import torch


def zadoff_chu_sequence(root, length, device):
    n = torch.arange(
        length,
        dtype=torch.float32,
        device=device,
    )

    phase = (
        -torch.pi
        * root
        * n
        * (n + 1)
        / length
    )

    return torch.exp(1j * phase)


#we use the pss sequence with N_id^(2) = 0 to simplify the implementation, since the other sequences are just cyclic shifts of this one 
def pss_sequence(device):
    x=torch.zeros(127, device=device, dtype=torch.float32)
    initial_state = torch.tensor([0,1,1,0,1,1,1], device=device, dtype=torch.float32)
    x[:7] = initial_state
    for i in range(127-7):
        x[i+7] = (x[i+4] + x[i]) % 2
    d_pss = 1-2*x

    return d_pss


def generate_pilot_sequence(device, sequence_type="PSS"):

    match sequence_type:
        case "Zadoff-Chu":
            s_z = zadoff_chu_sequence(
                root=13,
                length=11,
                device=device,
            )
            # s = [s_z; s_z; s_z; s_z; s_z]
            return s_z.repeat(5)

        case "PSS":
            d_pss = pss_sequence(device)
            return d_pss

        case _:
            raise ValueError(
                f"Unknown pilot sequence_type: {sequence_type!r}"
            )
