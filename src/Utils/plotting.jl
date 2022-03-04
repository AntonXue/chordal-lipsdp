
# Generate trajectories from a unit box
function randomTrajectories(N :: Int, ffnet :: NeuralNetwork)
  Random.seed!(1234)
  x1s = 2 * (rand(ffnet.xdims[1], N) .- 0.5) # Unit box
  # x1s = x1s ./ norm(x1s) # Unit vectors
  xfs = [runNetwork(x1s[:,k], ffnet) for k in 1:N]
  return xfs
end

# Plot some data to a file
function plotRandomTrajectories(N :: Int, ffnet :: NeuralNetwork, imgfile="~/Desktop/hello.png")
  # Make sure we can actually plot these in 2D
  @assert ffnet.xdims[end] == 2
  xfs = randomTrajectories(N, ffnet)
  d1s = [xf[1] for xf in xfs]
  d2s = [xf[2] for xf in xfs]
  p = scatter(d1s, d2s, markersize=2, alpha=0.3)
  savefig(p, imgfile)
end

# Plot different line data
# Get data of form (label1, ys1), (label2, ys2), ...
function plotLines(xs, labeled_lines :: Vector{Tuple{String, VecF64}};
                   title = "title", ylogscale = false, saveto = "~/Desktop/foo.png")
  # Make sure we have a consistent number of data
  @assert all(lys -> length(xs) == length(lys[2]), labeled_lines)
  plt = plot(title=title)
  colors = theme_palette(:auto)
  for (i, (lbl, ys)) in enumerate(labeled_lines)
    if ylogscale
      plot!(xs, ys, label=lbl, color=colors[i], yscale=:log10)
    else
      plot!(xs, ys, label=lbl, color=colors[i])
    end
  end
  savefig(plt, saveto)
  return plt
end

