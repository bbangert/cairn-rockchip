defmodule Cairn.Rockchip.MixProject do
  use Mix.Project

  @core_ref __DIR__ |> Path.join("CORE_REF") |> File.read!() |> String.trim()

  def project do
    [
      app: :cairn_rockchip,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      cairn_board: Cairn.Rockchip.Board,
      deps: deps(),
      # `test/attached` runs inside a core checkout (tools/attached-test.sh).
      test_load_filters: [&(String.ends_with?(&1, "_test.exs") and not attached?(&1))],
      test_ignore_filters: [&attached?/1],
      # `cairn_stage` is analysed as project code, not from the PLT: under
      # CAIRN_CORE_PATH it is a path dependency, and dialyxir rebuilds the
      # PLT on the lock and the app list, neither of which an edit in the
      # checkout touches. `:mix` for its board task. The PLTs sit outside
      # `priv`, which the board directory copies whole.
      dialyzer: [
        plt_add_apps: [:mix],
        plt_ignore_apps: [:cairn_stage],
        paths: Enum.map([:cairn_rockchip, :cairn_stage], &"_build/#{Mix.env()}/lib/#{&1}/ebin"),
        plt_core_path: "_build/plts",
        plt_local_path: "_build/plts"
      ]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp attached?(path), do: String.starts_with?(path, "test/attached/")

  # From core only `cairn_stage`, never `:cairn`: the board starts inside
  # `:cairn`'s start and reaches the base's libraries through the `base` map.
  defp deps do
    [
      cairn_stage(),
      {:membrane_h264_format, "~> 0.6.1"},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp cairn_stage do
    case System.get_env("CAIRN_CORE_PATH") do
      nil ->
        {:cairn_stage,
         git: "https://github.com/bbangert/cairn-nvr.git",
         ref: @core_ref,
         sparse: "libs/cairn_stage"}

      core ->
        {:cairn_stage, path: Path.join(core, "libs/cairn_stage")}
    end
  end
end
