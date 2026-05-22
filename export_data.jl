#!/usr/bin/env julia
# ===========================================================================
# export_data.jl  --  convert raw JutulDarcy .jld2 simulation data to HDF5.
#
#   Step 1 (ALWAYS run first):   julia export_data.jl --inspect
#       prints variable names + array sizes in each raw .jld2 file, so you can
#       confirm / correct the `raw:` section of configs/default.yaml.
#
#   Step 2:                      julia export_data.jl
#       decodes the flow state, pairs it with permeability, and writes
#       $data_dir/flow.h5  with datasets (as seen from Python / h5py):
#           saturation     (N, T, H, W)
#           permeability   (N, H, W)
#           gt_saturation  (T, H, W)
#           gt_permeability(H, W)
#
# Dimension convention: Julia is column-major, h5py is row-major. We build the
# Julia arrays in REVERSED dim order so h5py reads the shapes above directly,
# which also performs the (nx,nz) -> (H=nz, W=nx) image transpose for free.
# ===========================================================================
import Pkg
for p in ("JLD2", "HDF5", "YAML")
    try
        @eval import $(Symbol(p))
    catch
        Pkg.add(p); @eval import $(Symbol(p))
    end
end
using JLD2, HDF5, YAML

cfgpath = get(ENV, "NFS_CONFIG", joinpath(@__DIR__, "configs", "default.yaml"))
cfg = YAML.load_file(cfgpath)
P, R, G = cfg["paths"], cfg["raw"], cfg["grid"]

# ---------------------------------------------------------------------------
function inspect(path)
    println("\n=== ", path, " ===")
    if !isfile(path)
        println("  !! file not found"); return
    end
    jldopen(path, "r") do f
        for k in keys(f)
            v = try f[k] catch; "<unreadable>" end
            info = isa(v, AbstractArray) ? "$(size(v))  $(eltype(v))" : string(typeof(v))
            println("  ", rpad(k, 30), info)
        end
    end
end

if "--inspect" in ARGS
    for key in ("raw_flow_jld2", "raw_gt_jld2", "raw_perm_jld2")
        inspect(P[key])
    end
    println("\nNow set the `raw:` variable names in ", cfgpath, " and re-run without --inspect.")
    exit(0)
end

# ---------------------------------------------------------------------------
nx, nz, T = G["width"], G["height"], G["n_timesteps"]   # nx=512 (W), nz=256 (H)
np_ = nx * nz
satblk0 = R["saturation_block"]
mkpath(P["data_dir"])
outpath = joinpath(P["data_dir"], "flow.h5")

readvar(path, name) = jldopen(path, "r") do f
    haskey(f, name) || error("variable '$name' not found in $path  (run --inspect)")
    f[name]
end

# --- flow saturation -------------------------------------------------------
println("Reading flow state '", R["flow_var"], "' from ", P["raw_flow_jld2"])
state = readvar(P["raw_flow_jld2"], R["flow_var"])
N = size(state, 1)
println("  flow state size = ", size(state), "  ->  N = ", N, " simulations")

sat = Array{Float32}(undef, nx, nz, T, N)      # reversed dims for h5py
pres = Array{Float32}(undef, nx, nz, T, N)
for i in 1:N, k in 1:T
    osat = (satblk0 - 1 + k - 1) * np_
    opres = (satblk0 - 1 + T + k - 1) * np_
    sat[:, :, k, i]  = clamp.(reshape(Float32.(state[i, osat + 1 : osat + np_]), nx, nz), 0f0, 1f0)
    pres[:, :, k, i] = reshape(Float32.(state[i, opres + 1 : opres + np_]), nx, nz)
end
println("  decoded saturation + pressure -> Python shape (", N, ", ", T, ", ", nz, ", ", nx, ")")
state = nothing; GC.gc()        # free the ~15 GB raw state array

# --- ground-truth trajectory ----------------------------------------------
gt_sat = Array{Float32}(undef, nx, nz, T)
gt_pres = Array{Float32}(undef, nx, nz, T)
try
    gt = readvar(P["raw_gt_jld2"], R["gt_var"])
    gtrow = ndims(gt) == 1 ? gt : gt[1, :]
    for k in 1:T
        osat = (satblk0 - 1 + k - 1) * np_
        opres = (satblk0 - 1 + T + k - 1) * np_
        gt_sat[:, :, k]  = clamp.(reshape(Float32.(gtrow[osat + 1 : osat + np_]), nx, nz), 0f0, 1f0)
        gt_pres[:, :, k] = reshape(Float32.(gtrow[opres + 1 : opres + np_]), nx, nz)
    end
    println("  decoded ground-truth trajectory (saturation + pressure)")
catch e
    @warn "could not decode ground truth; writing zeros" exception=e
    fill!(gt_sat, 0f0); fill!(gt_pres, 0f0)
end

# --- permeability (selected from the (Nperm, nx, nz) pool by index) --------
perm = zeros(Float32, nx, nz, N)
gt_perm = zeros(Float32, nx, nz)
try
    pool = readvar(P["raw_perm_jld2"], R["perm_var"])       # (Nperm, nx, nz)
    ndims(pool) == 3 || error("expected a 3-D (Nperm, nx, nz) permeability pool")
    idx = Int.(readvar(P["raw_flow_jld2"], R["idx_var"]))   # one pool row per simulation
    for i in 1:N
        perm[:, :, i] = Float32.(pool[idx[i], :, :])
    end
    println("  paired permeability via '", R["idx_var"], "'  (", N,
            " sims, pool size ", size(pool, 1), ")")
    gtidx = readvar(P["raw_gt_jld2"], R["gt_idx_var"])
    gtidx = isa(gtidx, AbstractArray) ? Int(gtidx[1]) : Int(gtidx)
    gt_perm[:, :] = Float32.(pool[gtidx, :, :])
    println("  ground-truth permeability = ", R["perm_var"], "[", gtidx, "]")
catch e
    @warn "permeability not wired up - writing zeros (check raw.perm_var / idx_var)" exception=e
end

# --- write HDF5 ------------------------------------------------------------
isfile(outpath) && rm(outpath)
h5open(outpath, "w") do f
    f["saturation"]     = sat
    f["pressure"]       = pres
    f["permeability"]   = perm
    f["gt_saturation"]  = gt_sat
    f["gt_pressure"]    = gt_pres
    f["gt_permeability"] = gt_perm
    attributes(f)["layout"] = "dims reversed for h5py: saturation/pressure=(N,T,H,W)"
end
println("\nWrote ", outpath)
println("Done. Next:  python prepare_data.py --config ", cfgpath)
