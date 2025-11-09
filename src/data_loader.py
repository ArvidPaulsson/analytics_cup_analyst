"""
Core data loading utilities for match data.

This module provides functions to load match data from local files or GitHub,
including dynamic events, phases of play, tracking data, and match metadata.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import requests

try:  # Optional dependency for efficient preprocessing
    from kloppy import skillcorner
    from kloppy.domain import TrackingDataset

    KLOPPY_AVAILABLE = True
except ImportError:  # pragma: no cover - kloppy is optional
    skillcorner = None  # type: ignore[assignment]
    TrackingDataset = None  # type: ignore[assignment]
    KLOPPY_AVAILABLE = False


# GitHub base URLs
GITHUB_BASE_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/data"
GITHUB_MEDIA_BASE_URL = (
    "https://media.githubusercontent.com/media/SkillCorner/opendata/data"
)


def load_match_metadata(
    match_id: int, data_dir: Optional[Union[str, Path]] = None, use_github: bool = False
) -> Dict:
    """
    Load match metadata from JSON file.

    Parameters
    ----------
    match_id : int
        Match ID
    data_dir : str or Path, optional
        Local directory containing match data. If None, will try to infer from
        project structure.
    use_github : bool, default False
        If True, load from GitHub. If False, load from local files.

    Returns
    -------
    dict
        Match metadata including teams, players, periods, etc.
    """
    if use_github:
        url = f"{GITHUB_BASE_URL}/matches/{match_id}/{match_id}_match.json"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    else:
        if data_dir is None:
            # Try to infer data directory from project structure
            current_dir = Path(__file__).parent.parent
            data_dir = current_dir.parent / "opendata" / "data"

        data_dir = Path(data_dir)
        match_file = data_dir / "matches" / str(match_id) / f"{match_id}_match.json"

        if not match_file.exists():
            raise FileNotFoundError(
                f"Match file not found: {match_file}. "
                f"Set use_github=True to load from GitHub."
            )

        with open(match_file, "r") as f:
            return json.load(f)


def load_dynamic_events(
    match_id: int, data_dir: Optional[Union[str, Path]] = None, use_github: bool = False
) -> pd.DataFrame:
    """
    Load dynamic events data from CSV file.

    Parameters
    ----------
    match_id : int
        Match ID
    data_dir : str or Path, optional
        Local directory containing match data
    use_github : bool, default False
        If True, load from GitHub. If False, load from local files.

    Returns
    -------
    pd.DataFrame
        Dynamic events data
    """
    if use_github:
        url = f"{GITHUB_BASE_URL}/matches/{match_id}/{match_id}_dynamic_events.csv"
        return pd.read_csv(url)
    else:
        if data_dir is None:
            current_dir = Path(__file__).parent.parent
            data_dir = current_dir.parent / "opendata" / "data"

        data_dir = Path(data_dir)
        events_file = (
            data_dir / "matches" / str(match_id) / f"{match_id}_dynamic_events.csv"
        )

        if not events_file.exists():
            raise FileNotFoundError(
                f"Dynamic events file not found: {events_file}. "
                f"Set use_github=True to load from GitHub."
            )

        return pd.read_csv(events_file, low_memory=False)


def load_phases_of_play(
    match_id: int, data_dir: Optional[Union[str, Path]] = None, use_github: bool = False
) -> pd.DataFrame:
    """
    Load phases of play data from CSV file.

    Parameters
    ----------
    match_id : int
        Match ID
    data_dir : str or Path, optional
        Local directory containing match data
    use_github : bool, default False
        If True, load from GitHub. If False, load from local files.

    Returns
    -------
    pd.DataFrame
        Phases of play data
    """
    if use_github:
        url = f"{GITHUB_BASE_URL}/matches/{match_id}/{match_id}_phases_of_play.csv"
        return pd.read_csv(url)
    else:
        if data_dir is None:
            current_dir = Path(__file__).parent.parent
            data_dir = current_dir.parent / "opendata" / "data"

        data_dir = Path(data_dir)
        phases_file = (
            data_dir / "matches" / str(match_id) / f"{match_id}_phases_of_play.csv"
        )

        if not phases_file.exists():
            raise FileNotFoundError(
                f"Phases of play file not found: {phases_file}. "
                f"Set use_github=True to load from GitHub."
            )

        return pd.read_csv(phases_file, low_memory=False)


def load_tracking_data(
    match_id: int,
    data_dir: Optional[Union[str, Path]] = None,
    use_github: bool = True,
    *,
    coordinates: str = "skillcorner",
    sample_rate: Optional[float] = None,
    limit: Optional[int] = None,
    orientation: Optional[str] = "STATIC_HOME_AWAY",
    engine: str = "pandas",
    **kwargs,
) -> pd.DataFrame:
    """
    Load tracking data via Kloppy and return a dataframe in the requested engine.

    Parameters
    ----------
    match_id : int
        Match ID
    data_dir : str or Path, optional
        Local directory containing match data (only used when use_github=False)
    use_github : bool, default True
        Stream open data from GitHub (recommended).
    coordinates : str, default 'skillcorner'
        Coordinate system passed to Kloppy.
    sample_rate : float, optional
        Downsampling factor (e.g., 0.5 to halve FPS).
    limit : int, optional
        Limit number of frames for faster prototyping.
    orientation : str, optional
        Orientation transform passed to Kloppy.
    engine : str, default 'pandas'
        Engine used in `dataset.to_df`.
    kwargs : dict
        Additional options forwarded to Kloppy.
    """
    if not KLOPPY_AVAILABLE:
        raise ImportError(
            "Kloppy is required for tracking operations. "
            "Install it with `pip install kloppy>=3.18.0`."
        )
    dataset = load_tracking_dataset_kloppy(
        match_id,
        coordinates=coordinates,
        sample_rate=sample_rate,
        limit=limit,
        to_orientation=orientation,
        use_github=use_github,
        data_dir=data_dir,
        **kwargs,
    )

    df = dataset.to_df(engine=engine)  # type: ignore[attr-defined]
    if engine == "polars":
        df = df.to_pandas()  # type: ignore[call-arg]
    return df  # type: ignore[return-value]


def load_tracking_dataset_kloppy(
    match_id: int,
    *,
    coordinates: str = "skillcorner",
    sample_rate: Optional[float] = None,
    limit: Optional[int] = None,
    to_orientation: Optional[str] = None,
    use_github: bool = True,
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> "TrackingDataset":
    """
    Load SkillCorner tracking data using Kloppy's optimized loader.

    Parameters
    ----------
    match_id : int
        Match ID
    coordinates : str, default 'skillcorner'
        Coordinate system to use when loading data
    sample_rate : float, optional
        Downsampling factor (e.g., 0.5 to halve the frame rate)
    limit : int, optional
        Limit number of frames loaded
    to_orientation : str, optional
        Optional orientation transformation applied after loading
    use_github : bool, default True
        If True, stream open data from GitHub via Kloppy.
        If False, falls back to local files (GitHub is recommended for Kloppy).
    data_dir : str or Path, optional
        Local data directory. Only used when use_github is False.
    kwargs : dict
        Additional keyword arguments forwarded to Kloppy.

    Returns
    -------
    TrackingDataset
        Kloppy tracking dataset
    """
    if not KLOPPY_AVAILABLE:
        raise ImportError(
            "Kloppy is not installed. Install it with `pip install kloppy>=3.18.0`."
        )

    if use_github:
        dataset = skillcorner.load_open_data(  # type: ignore[call-arg]
            match_id=match_id,
            coordinates=coordinates,
            sample_rate=sample_rate,
            limit=limit,
            **kwargs,
        )
    else:
        if data_dir is None:
            current_dir = Path(__file__).parent.parent
            match_dir = (
                current_dir.parent / "opendata" / "data" / "matches" / str(match_id)
            )
        else:
            match_dir = Path(data_dir) / "matches" / str(match_id)

        tracking_file = match_dir / f"{match_id}_tracking_extrapolated.jsonl"
        metadata_file = match_dir / f"{match_id}_match.json"

        if not tracking_file.exists():
            raise FileNotFoundError(
                f"Tracking data file not found: {tracking_file}. "
                "Set use_github=True to stream via GitHub, or ensure Kloppy can access the local file."
            )

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Match metadata file not found: {metadata_file}. "
                "Kloppy requires metadata alongside tracking data."
            )

        try:
            dataset = skillcorner.load_tracking_data(  # type: ignore[attr-defined]
                file_path=str(tracking_file),
                metadata=str(metadata_file),
                coordinates=coordinates,
                sample_rate=sample_rate,
                limit=limit,
                **kwargs,
            )
        except AttributeError as exc:  # pragma: no cover - depends on Kloppy version
            raise RuntimeError(
                "The installed Kloppy version does not expose `skillcorner.load_tracking_data`. "
                "Either upgrade Kloppy to >=3.18.0 or set `use_github=True` to stream open data."
            ) from exc

    if to_orientation:
        dataset = dataset.transform(to_orientation=to_orientation)

    return dataset


def load_match_data(
    match_id: int,
    data_dir: Optional[Union[str, Path]] = None,
    use_github: bool = False,
    load_tracking: bool = True,
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Load all data for a match in one call.

    Parameters
    ----------
    match_id : int
        Match ID
    data_dir : str or Path, optional
        Local directory containing match data
    use_github : bool, default False
        If True, load from GitHub. If False, load from local files.
    load_tracking : bool, default True
        If True, load tracking data. If False, skip tracking data loading
        (useful if tracking data is large and not needed).

    Returns
    -------
    dict
        Dictionary containing:
        - 'metadata': Match metadata (dict)
        - 'dynamic_events': Dynamic events DataFrame
        - 'phases_of_play': Phases of play DataFrame
        - 'tracking_data': Tracking data DataFrame (if load_tracking=True)
    """
    data = {
        "metadata": load_match_metadata(match_id, data_dir, use_github),
        "dynamic_events": load_dynamic_events(match_id, data_dir, use_github),
        "phases_of_play": load_phases_of_play(match_id, data_dir, use_github),
    }

    if load_tracking:
        dataset = load_tracking_dataset_kloppy(
            match_id,
            use_github=use_github,
            data_dir=data_dir,
        )
        tracking_df = dataset.to_df(engine="pandas")  # type: ignore[attr-defined]
        rename_candidates = {
            "frame_id": "frame",
            "frame.id": "frame",
            "frame_timestamp": "timestamp",
            "frame.timestamp": "timestamp",
            "player_id": "player_id",
            "player.id": "player_id",
            "team_id": "team_id",
            "player.team_id": "team_id",
            "player.team.id": "team_id",
            "coordinates.x": "x",
            "coordinates.y": "y",
            "coordinates.z": "z",
            "player_coordinates.x": "x",
            "player_coordinates.y": "y",
            "player_coordinates.z": "z",
            "velocity": "speed",
            "ball_x": "ball_x",
            "ball_y": "ball_y",
            "ball_z": "ball_z",
            "ball.coordinates.x": "ball_x",
            "ball.coordinates.y": "ball_y",
            "ball.coordinates.z": "ball_z",
        }
        tracking_df = tracking_df.rename(
            columns={
                src: dst
                for src, dst in rename_candidates.items()
                if src in tracking_df.columns
            }
        )
        data["tracking_dataset"] = dataset

        required_columns = [
            "frame",
            "timestamp",
            "player_id",
            "team_id",
            "x",
            "y",
            "z",
            "speed",
            "ball_x",
            "ball_y",
            "ball_z",
        ]
        for col in required_columns:
            if col not in tracking_df.columns:
                tracking_df[col] = None

        data["tracking_data"] = tracking_df

    return data


def load_all_matches_metadata(
    data_dir: Optional[Union[str, Path]] = None, use_github: bool = False
) -> pd.DataFrame:
    """
    Load metadata for all available matches.

    Parameters
    ----------
    data_dir : str or Path, optional
        Local directory containing match data
    use_github : bool, default False
        If True, load from GitHub. If False, load from local files.

    Returns
    -------
    pd.DataFrame
        DataFrame with match IDs and basic information
    """
    if use_github:
        url = f"{GITHUB_BASE_URL}/matches.json"
        response = requests.get(url)
        response.raise_for_status()
        matches = response.json()
        return pd.DataFrame(matches)
    else:
        if data_dir is None:
            current_dir = Path(__file__).parent.parent
            data_dir = current_dir.parent / "opendata" / "data"

        data_dir = Path(data_dir)
        matches_file = data_dir / "matches.json"

        if not matches_file.exists():
            raise FileNotFoundError(
                f"Matches file not found: {matches_file}. "
                f"Set use_github=True to load from GitHub."
            )

        with open(matches_file, "r") as f:
            matches = json.load(f)

        return pd.DataFrame(matches)
