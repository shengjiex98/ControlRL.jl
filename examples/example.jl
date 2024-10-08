### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 04f3e318-6489-11ef-1ccc-89d0ed2d1029
begin
	import Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
	using Plots
	plotlyjs()
	using LinearAlgebra
	using PlutoUI
	push!(LOAD_PATH, "$(@__DIR__)/../src")
	using ControlRL

	TableOfContents()
end

# ╔═╡ abb82687-12f0-4754-b3dd-49c03f3c9ad1
md"""
## Setting up Packages
"""

# ╔═╡ 5ff9a070-12a3-45fd-a1b4-b619998e468a
md"""
## Defining the policy π

The reference policy π₀ is defined as meeting all deadlines:
"""

# ╔═╡ 2fd32b80-1d52-4a97-8dab-46f1278ebc58
π₀(actions::Vector{Bool}, states::Vector{Vector{Float64}}, rewards::Vector{Float64}) = true

# ╔═╡ 05117c1b-a715-44f4-9f98-c7fbca48ce30
md"""
**You only need to change the section below.**

Define your own policy here.
The reward is currently defined as the negation of distance between current
state and the state obtained by following the reference policy π₀.

Several example policies are provided. Feel free to experiment with one of them,
combine two or more of them, or experiment with your own policies!
"""

# ╔═╡ fac6522b-c189-4b7c-be35-11d038a854bc
"""
	π(actions, states, rewards)

The policy π decides the action (deadline hit/miss) based on the history of
past actions, states, and rewards. It returns a single true/false value.
"""
function π(actions::Vector{Bool}, states::Vector{Vector{Float64}}, rewards::Vector{Float64})
	## Example 1: return true when the distance to reference is greater than 1
	# return rewards[end] <= -1
	
	## Example 2: return true when the distance increase consecutively for 3 steps
	# return length(rewards) > 2 && (rewards[end-2] > rewards[end-1] > rewards[end])

	## Example 3: return true when the utilization up to now is less than 0.5
	return length(actions) > 0 && (sum(actions) / length(actions) < 0.5)
end

# ╔═╡ 8015eee9-cfe9-4890-a3a5-e3f97517d1d2
md"""
!!! note

	Some of the policies are defined with the [compact function declaration syntax](https://docs.julialang.org/en/v1/manual/functions/#man-functions), which is equivalent to the normal syntax. I.e., the following two declarations are equivalent:
	```julia
	π₀(actions::Vector{Bool}, states::Vector{Vector{Float64}}, rewards::Vector{Float64}) = true
	```
	and
	```julia
	function π₀(actions::Vector{Bool}, states::Vector{Vector{Float64}}, rewards::Vector{Float64})
		return true
	end
	```
!!! note
	The symbol "π" can be entered by typing `\pi` followed by the `<tab>` key. This is a nice feature of Julia and can be used to enter [any UNICODE character](https://docs.julialang.org/en/v1/manual/unicode-input/)! Under the hood, this is possible because Julia allows UNICODE characters to be part of function names and variable names in addition to the ASCII characters.
"""

# ╔═╡ c57331ac-ea8e-44b9-9213-9cad71788244
md"""
## Outputs

Define the sampling period $T$ and time horizon $H$
"""

# ╔═╡ 311337a7-9402-4ade-8707-e9ba9b72fa3b
const T = 0.02

# ╔═╡ 837c7cf2-66cb-4ebe-a417-df1e5da9cc73
const H = 500

# ╔═╡ 2fee3edd-bba5-4738-a43c-691c9b4cf3de
md"""
Display all available models.
"""

# ╔═╡ 64bf8494-2e6c-4ad7-bffa-cf4a6ae74456
benchmarks

# ╔═╡ 84f2e971-241e-4895-9daf-18a65de77ea5
md"""
Convert a chosen model from continuous to discrete form using sampling period $T$ = $T
"""

# ╔═╡ a274cbd6-13c3-499f-8407-41211e25dae1
sys = c2d(benchmarks[:CC], T)

# ╔═╡ 4730f9cd-493d-42ea-9006-013a746a149f
md"""
Initiate the simulation environment and Simulate the system for $H$ steps.
"""

# ╔═╡ cbd51753-b04a-40c9-8f5b-9b88c74fc6f7
actions, states, rewards, ideal_states = let 
	env = Environment(sys)
	sim!(env, π, H)
end

# ╔═╡ 09e3258a-a3fe-43a0-8f65-0d4a13b5c513
md"""
Calculate the total rewards over $H=$ $H steps
"""

# ╔═╡ 45b657ae-6220-44b3-aa35-cae27bf730b0
total_rewards = sum(rewards)

# ╔═╡ eb39929a-46b6-4d53-a3e9-30af17500b54
md"""
Plot the trajectory of system dynamics following the reference policy.
"""

# ╔═╡ d9b853f3-4f79-487e-98ef-2f1b321b9d69
md"""
Plot the trajectory of system dynamics following the defined policy.
"""

# ╔═╡ ad589207-0f42-446a-b3d6-167a54c02950
md"""
Plot the rewards (distance to the reference trajectory).
"""

# ╔═╡ e705f0df-bdcc-4673-976f-5d096c6a2abc
md"""
## Helper functions
"""

# ╔═╡ 5402045e-83ef-4b90-9f14-1796fdadacc0
"""
	plotH(x)

Plot the system states as recorded in states matrix.
"""
function plotH(states::Matrix{Float64})
	# Transposing `states` because plot uses the first dimension as indices.
	plot(0:size(states, 2) - 1, states')
end

# ╔═╡ c34375b4-70ab-4468-9186-c43948f22a7a
plotH(v::Vector{Float64}) = plotH(reshape(v, 1, :))

# ╔═╡ 19634c38-ec2c-45a0-862a-3f92beecb5a1
plotH(ideal_states)

# ╔═╡ 66fbe369-33e7-409f-ae66-29c1984518b3
plotH(states)

# ╔═╡ 7c54f79a-6bb4-4190-ab13-bca71bb868a8
plotH(rewards)

# ╔═╡ Cell order:
# ╟─abb82687-12f0-4754-b3dd-49c03f3c9ad1
# ╠═04f3e318-6489-11ef-1ccc-89d0ed2d1029
# ╟─5ff9a070-12a3-45fd-a1b4-b619998e468a
# ╠═2fd32b80-1d52-4a97-8dab-46f1278ebc58
# ╟─05117c1b-a715-44f4-9f98-c7fbca48ce30
# ╠═fac6522b-c189-4b7c-be35-11d038a854bc
# ╟─8015eee9-cfe9-4890-a3a5-e3f97517d1d2
# ╟─c57331ac-ea8e-44b9-9213-9cad71788244
# ╠═311337a7-9402-4ade-8707-e9ba9b72fa3b
# ╠═837c7cf2-66cb-4ebe-a417-df1e5da9cc73
# ╟─2fee3edd-bba5-4738-a43c-691c9b4cf3de
# ╠═64bf8494-2e6c-4ad7-bffa-cf4a6ae74456
# ╟─84f2e971-241e-4895-9daf-18a65de77ea5
# ╠═a274cbd6-13c3-499f-8407-41211e25dae1
# ╟─4730f9cd-493d-42ea-9006-013a746a149f
# ╠═cbd51753-b04a-40c9-8f5b-9b88c74fc6f7
# ╟─09e3258a-a3fe-43a0-8f65-0d4a13b5c513
# ╠═45b657ae-6220-44b3-aa35-cae27bf730b0
# ╟─eb39929a-46b6-4d53-a3e9-30af17500b54
# ╠═19634c38-ec2c-45a0-862a-3f92beecb5a1
# ╟─d9b853f3-4f79-487e-98ef-2f1b321b9d69
# ╠═66fbe369-33e7-409f-ae66-29c1984518b3
# ╟─ad589207-0f42-446a-b3d6-167a54c02950
# ╠═7c54f79a-6bb4-4190-ab13-bca71bb868a8
# ╟─e705f0df-bdcc-4673-976f-5d096c6a2abc
# ╠═5402045e-83ef-4b90-9f14-1796fdadacc0
# ╠═c34375b4-70ab-4468-9186-c43948f22a7a
