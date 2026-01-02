import torch
import torch.nn as nn
import click
from utils.svd_logger import SVDLogger, get_logger_from_env

class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None, name=None, sigma_fuse='UV', V_transpose=True, succinct=False, succinct_split="A") -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=False)

        self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
        self.truncation_rank = S.size(0)

        self.succinct_split = succinct_split
        self.sigma_fuse = sigma_fuse
        
        self.name = name

        if V_transpose:
            self.ALinear.weight.data = U
            self.BLinear.weight.data = V.T
        else:
            self.ALinear.weight.data = U
            self.BLinear.weight.data = V

        if succinct:
            # Decide which part to split
            C_A = torch.linalg.cond(self.ALinear.weight.data[:self.truncation_rank, :self.truncation_rank])
            C_B = torch.linalg.cond(self.BLinear.weight.data[:self.truncation_rank, :self.truncation_rank])
            
            print(f"C_A: {float(C_A)}, C_B: {float(C_B)}")

            if (float(C_A) < float(C_B)):
                self.succinct_split = "A"
            else:
                self.succinct_split = "B"
            
            if self.succinct_split == "A":
                self.sigma_fuse = "V"
            elif self.succinct_split == "B":
                self.sigma_fuse = "U"
        
        # Fuse the singular values
        if self.sigma_fuse == 'UV':
            self.ALinear.weight.data = self.ALinear.weight.data.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = self.BLinear.weight.data.mul(S.sqrt().view(-1, 1)).contiguous()
        elif self.sigma_fuse == 'U':
            self.ALinear.weight.data = self.ALinear.weight.data.mul(S).contiguous()
            self.BLinear.weight.data = self.BLinear.weight.data.contiguous()
        elif self.sigma_fuse == 'V':
            self.ALinear.weight.data = self.ALinear.weight.data.contiguous()
            self.BLinear.weight.data = self.BLinear.weight.data.mul(S.view(-1, 1)).contiguous()
        else:
            raise ValueError("sigma_fuse should be 'UV', 'U' or 'V'")

        # Post processing for succinct SVD
        if succinct:
            click.secho(f"Succinct", fg="yellow")
            self.succinct = True
            
            if self.succinct_split == "A":

                C = self.ALinear.weight.data[:self.truncation_rank, :self.truncation_rank]
                self.BLinear.weight.data = torch.matmul(C.to(torch.float64), self.BLinear.weight.data.to(torch.float64)).to(torch.float32)
                
                C_inv = torch.linalg.inv(C.to(torch.float64))
                assert C_inv.dtype == torch.float64
                
                U_low_rank_fused_C_inv = torch.matmul(self.ALinear.weight.data.to(torch.float64), C_inv).to(torch.float32)
                
                # Update the weight of ALinear and BLinear
                self.ALinear.weight.data = U_low_rank_fused_C_inv[self.truncation_rank:, :] # (m-r, r)

                # Stop compressing if there is inf in the weight
                if torch.isinf(self.BLinear.weight).any():
                    click.secho(f"inf in BLinear", fg="red")
                    exit()
                
                if torch.isinf(self.ALinear.weight).any():
                    click.secho(f"inf in ALinear", fg="red")
                    exit()
                
                print(self.succinct_split, self.sigma_fuse, self.ALinear.weight.shape, self.BLinear.weight.shape, self.truncation_rank)
                
                del C, C_inv, U_low_rank_fused_C_inv
            
            elif self.succinct_split == "B":
                C = self.BLinear.weight.data[:self.truncation_rank, :self.truncation_rank]
                
                C_inv = torch.linalg.inv(C.to(torch.float64))
                assert C_inv.dtype == torch.float64
                
                C_inv_V = torch.matmul(C_inv, self.BLinear.weight.data.to(torch.float64)).to(torch.float32)
                
                # Update the weight of ALinear and BLinear
                self.ALinear.weight.data = torch.matmul(self.ALinear.weight.data.to(torch.float64), C.to(torch.float64)).to(torch.float32)
                self.BLinear.weight.data = C_inv_V[:, self.truncation_rank:] #(r, d-r)

                click.secho(f"ALinear: min={torch.min(self.ALinear.weight.data)}, max={torch.max(self.ALinear.weight.data)}", fg="blue")
                click.secho(f"BLinear: min={torch.min(self.BLinear.weight.data)}, max={torch.max(self.BLinear.weight.data)}", fg="blue")

                # Stop the compression if there is inf in the weight
                if torch.isinf(self.BLinear.weight).any():
                    click.secho(f"inf in BLinear", fg="red")
                    exit()
                
                if torch.isinf(self.ALinear.weight).any():
                    click.secho(f"inf in ALinear", fg="red")
                    exit()
            

                assert self.ALinear.weight.dtype == torch.float32
                assert self.BLinear.weight.dtype == torch.float32

                print(self.succinct_split, self.sigma_fuse, self.ALinear.weight.shape, self.BLinear.weight.shape, self.truncation_rank)

                del C, C_inv_V, C_inv
                
                print(self.ALinear.weight.shape, self.BLinear.weight.shape, self.truncation_rank)
            else:
                raise ValueError("succinct_split should be 'A' or 'B'")

            # Check the range of weight
            click.secho(f"ALinear: min={torch.min(self.ALinear.weight.data)}, max={torch.max(self.ALinear.weight.data)}", fg="yellow")
            click.secho(f"BLinear: min={torch.min(self.BLinear.weight.data)}, max={torch.max(self.BLinear.weight.data)}", fg="yellow")
            if abs(torch.max(self.ALinear.weight.data)) > 65536 or abs(torch.max(self.BLinear.weight.data)) > 65536:
                click.secho("Overflow in FP16!!", fg="red")
            
            if (self.ALinear.weight.data != self.ALinear.weight.data).any():
                click.secho(f"nan in ALinear", fg="red")
            if (self.BLinear.weight.data != self.BLinear.weight.data).any():
                click.secho(f"nan in BLinear", fg="red")

            print("-------------------------------------------------------------------------")
            
        else:
            self.succinct = False
            
            # Check the range of weight
            click.secho(f"ALinear: min={torch.min(self.ALinear.weight.data)}, max={torch.max(self.ALinear.weight.data)}", fg="yellow")
            click.secho(f"BLinear: min={torch.min(self.BLinear.weight.data)}, max={torch.max(self.BLinear.weight.data)}", fg="yellow")
            if abs(torch.max(self.ALinear.weight.data)) > 65536 or abs(torch.max(self.BLinear.weight.data)) > 65536:
                click.secho("Overflow in FP16!!", fg="red")
            
            print("-------------------------------------------------------------------------")

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1,
        sigma_fuse="UV"
    ):
        if param_ratio >= 1:
            print(max(linear.in_features, linear.out_features))
            return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        print(rank)
        # print("rank", rank)
        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division
            w = w * scaling_diag_matrix.view(1, -1)
        Us = []
        Ss = []
        Vs = []
        # try:
        #     # torch.manual_seed(0)
        #     # U, S, V = torch.svd_lowrank(w, q=rank)
            
        #     U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        # except:
        #     print(f"svd failed for {linear}, disable act_aware")
        #     return (
        #         nn.Linear(linear.in_features, linear.out_features)
        #         .to(linear.weight.dtype)
        #         .to(linear.weight.device)
        #     )
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        # Low rank approximation
        U = U[:, 0:rank]
        S = S[0:rank]
        V = Vh[0:rank, :]
        V.transpose_(0, 1)
        
        if act_aware:
            V = V / scaling_diag_matrix.view(-1, 1)
        Us = [U]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        for S in Ss:
            if (S!=S).any():
                print("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features)
                    .to(linear.weight.dtype)
                    .to(linear.weight.device)
                )
        for U in Us:
            if (U!=U).any():
                print("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features)
                    .to(linear.weight.dtype)
                    .to(linear.weight.device)
                )
        for V in Vs:
            if (V!=V).any():
                print("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features)
                    .to(linear.weight.dtype)
                    .to(linear.weight.device)
                )

        assert len(Us) == len(Ss) == len(Vs) == 1
        new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse)
        return new_linear.to(linear.weight.dtype)

    @staticmethod
    def from_linear_rank(
        linear: nn.Linear,
        name: str,
        rank: int,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=0.5,
        succinct=False,
        sigma_fuse="UV"
    ):
        full_rank = min(linear.in_features, linear.out_features)
        if rank >= full_rank:
            print(full_rank)
            return linear
        
        print(rank)

        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # print("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
                # scaling_diag_matrix *= linear.scaling_diag_matrix**0.5
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix *= linear.fisher_info**alpha
                # scaling_diag_matrix *= linear.fisher_info**1
            # if not (scaling_diag_matrix == scaling_diag_matrix).all():
            #     breakpoint()
            scaling_diag_matrix += 1e-6  # avoid zero division
            w = w * scaling_diag_matrix.view(1, -1)
        Us = []
        Ss = []
        Vs = []
        # try:
        #     # torch.manual_seed(0)
        #     # U, S, V = torch.svd_lowrank(w, q=rank)
        #     U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        # except:
        #     print(f"svd failed for {linear}, disable act_aware")
        #     return (
        #         nn.Linear(linear.in_features, linear.out_features)
        #         .to(linear.weight.dtype)
        #         .to(linear.weight.device)
        #     )
        
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        
        #record svd energy in 
        log_svd_energy("SVD-1", S)
        U2, S2, V2h = torch.linalg.svd(U, full_matrices=False)
        log_svd_energy("SVD-U", S2)

        U3, S3, V3h = torch.linalg.svd(Vh, full_matrices=False)
        log_svd_energy("SVD-V", S3)


        # Low rank approximation
        U = U[:, 0:rank]
        S = S[0:rank]
        V = Vh[0:rank, :]
        V.transpose_(0, 1)
        
        if act_aware:
            V = V / scaling_diag_matrix.view(-1, 1)

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        new_linear = SVDLinear(U, S, V, bias, name=name, sigma_fuse=sigma_fuse, V_transpose=True, succinct=succinct)

        return new_linear.to(linear.weight.dtype)
    
    @staticmethod
    def from_linear_whiten(
        linear: nn.Linear,
        param_ratio: float,
        sigma_fuse='UV'
    ):
        if param_ratio >= 1:
            full_rank = min(linear.in_features, linear.out_features)
            print(full_rank)
            return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        
        rank = compressed_params // (linear.in_features + linear.out_features)
        print(rank)
        # print("rank", rank)
        w = linear.weight.data.float()
        
        H, W = w.size()

        try:
            scaling_diag_matrix = linear.scaling_diag_matrix.to(w.device)
        except AttributeError:
            raise FileExistsError("Cache may not be loaded correctly")
        
        # Get the inverse of scaling_diag_matrix
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

        # Multiply scaling_diag_matrix to weight matrix
        W_scale = torch.matmul(w, scaling_diag_matrix.to(torch.float32))
        
        U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
        
        V = torch.matmul(Vt, scaling_matrix_inv)
        
        # Low rank approximation to the target rank
        U = U[:, :rank]
        S = S[:rank]
        V = V[:rank, :]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        if (S!=S).any():
            print("nan in S")
            return (
                nn.Linear(linear.in_features, linear.out_features)
                .to(linear.weight.dtype)
                .to(linear.weight.device)
                )
        if (U!=U).any():
            print("nan in U")
            return (
                nn.Linear(linear.in_features, linear.out_features)
                .to(linear.weight.dtype)
                .to(linear.weight.device)
            )
        if (V!=V).any():
            print("nan in V")
            return (
                nn.Linear(linear.in_features, linear.out_features)
                .to(linear.weight.dtype)
                .to(linear.weight.device)
            )

        new_linear = SVDLinear(U, S, V, bias, V_transpose=False, sigma_fuse=sigma_fuse)
        return new_linear.to(linear.weight.dtype)
    
    @staticmethod
    def from_linear_whiten_rank(
        linear: nn.Linear,
        name: str,
        rank: int,
        succinct=False,
        sigma_fuse='UV'
    ):
        full_rank = min(linear.in_features, linear.out_features)
        if rank >= full_rank:
            print(full_rank, rank)
            return linear
        
        print(rank)
        # print("rank", rank)
        w = linear.weight.data.float()
        
        H, W = w.size()

        try:
            scaling_diag_matrix = linear.scaling_diag_matrix.to(w.device)
        except AttributeError:
            raise FileExistsError("Cache may not be loaded correctly")
        
        # Get the inverse of scaling_diag_matrix
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

        # Multiply scaling_diag_matrix to weight matrix
        W_scale = torch.matmul(w, scaling_diag_matrix.to(torch.float32))
        
        U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
        
        V = torch.matmul(Vt, scaling_matrix_inv)
        
        # Low rank approximation to the target rank
        U = U[:, :rank]
        S = S[:rank]
        V = V[:rank, :]
        
        try:
            logger = get_logger_from_env()
            if logger is not None:
                m = linear.out_features
                n = linear.in_features
                logger.log(
                    layer=name,
                    w_name=name.split(".")[-1],
                    m=m, n=n, rank=int(rank),
                    S=S, U=U, V=V,
                    svd_iter=None,
                    extra={"method":"whiten"}
                )
        except Exception as e:
            print("[SVDLOG] from_linear_whiten_rank log failed:", e)

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        if (S!=S).any():
            print("nan in S")
            return (
                nn.Linear(linear.in_features, linear.out_features)
                .to(linear.weight.dtype)
                .to(linear.weight.device)
                )
        if (U!=U).any():
            print("nan in U")
            return (
                nn.Linear(linear.in_features, linear.out_features)
                .to(linear.weight.dtype)
                .to(linear.weight.device)
            )
        if (V!=V).any():
            print("nan in V")
            return (
                nn.Linear(linear.in_features, linear.out_features)
                .to(linear.weight.dtype)
                .to(linear.weight.device)
            )
    
        new_linear = SVDLinear(U, S, V, bias, name=name, sigma_fuse=sigma_fuse, V_transpose=False, succinct=succinct)
        return new_linear.to(linear.weight.dtype)
    

    def forward(self, inp):
        if inp.numel() > 0:
            mx_abs = inp.abs().max()
            if mx_abs > 65536:
                click.secho(f"Max: {mx_abs} Overflow in FP16!!", fg="red")
                inp = inp / mx_abs
        if (inp != inp).any():
            click.secho(f"NaN in {self.name}", fg="red")

        # Cast input to FP16
        if self.succinct:
            # inp suppose to be 3D tensor (batch, seq_len, hidden_dim)
            if self.succinct_split == "A":
                M_1 = self.BLinear(inp) # (B, N, r)
                M_2 = self.ALinear(M_1) # (B, N, m-r)
                y = torch.cat((M_1, M_2), dim=-1)
            elif self.succinct_split == "B":
                X1 = inp[:, :, :self.truncation_rank] 
                X2 = inp[:, :, self.truncation_rank:]
                M_1 = self.BLinear(X2)
                y = self.ALinear(X1 + M_1)
            else:
                raise ValueError("succinct_split should be 'A' or 'B'")
        else:
            y = self.BLinear(inp)
            y = self.ALinear(y)
        
        return y
