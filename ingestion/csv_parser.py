"""
CSV parsing for keyword uploads.
Supports any CSV format with column selection.
"""
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple, Optional, Any
from io import BytesIO, StringIO

from core.exceptions import ValidationError


class CSVParser:
    """
    Parses CSV files containing keywords.
    Supports any CSV format - user can select keyword column.
    """
    
    # Common column name variations for auto-detection
    KEYWORD_COLUMN_NAMES = [
        "keyword", "keywords", "query", "queries",
        "search term", "search_term", "searchterm",
        "term", "terms", "keyphrase", "key phrase",
        "kw", "word", "words", "top queries"
    ]
    
    VOLUME_COLUMN_NAMES = [
        "search volume", "searchvolume", "search_volume",
        "volume", "vol", "monthly searches", "monthly_searches",
        "searches", "avg_monthly_searches", "avg monthly searches",
        "msv", "sv", "impressions", "clicks"
    ]
    
    KD_COLUMN_NAMES = [
        "keyword difficulty", "keyword_difficulty", "kd",
        "difficulty", "seo difficulty", "seo_difficulty",
        "competition", "comp", "position", "ctr"
    ]
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.original_columns: List[str] = []
        self.keyword_column: Optional[str] = None
        self.volume_column: Optional[str] = None
        self.kd_column: Optional[str] = None
    
    def preview(
        self,
        file_content: BytesIO,
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """
        Preview CSV file and return column information.
        
        Args:
            file_content: File content as BytesIO
            encoding: File encoding (default: utf-8)
        
        Returns:
            Dict with columns, sample data, and auto-detected columns
        """
        try:
            for enc in [encoding, "utf-8", "latin-1", "cp1252"]:
                try:
                    file_content.seek(0)
                    self.df = pd.read_csv(
                        file_content,
                        encoding=enc,
                        low_memory=False,
                        on_bad_lines='skip'  # Skip malformed rows
                    )
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValidationError(
                    "Could not decode file with any supported encoding"
                )
        except Exception as e:
            raise ValidationError(f"Failed to read CSV: {str(e)}")
        
        # Store original column names
        self.original_columns = list(self.df.columns)
        
        # Normalize for detection
        normalized = {
            str(col).lower().strip(): col
            for col in self.df.columns
        }
        
        # Auto-detect keyword column
        detected_kw = None
        for name in self.KEYWORD_COLUMN_NAMES:
            if name in normalized:
                detected_kw = normalized[name]
                break
        
        # Auto-detect volume column
        detected_vol = None
        for name in self.VOLUME_COLUMN_NAMES:
            if name in normalized:
                detected_vol = normalized[name]
                break
        
        # Auto-detect KD column
        detected_kd = None
        for name in self.KD_COLUMN_NAMES:
            if name in normalized:
                detected_kd = normalized[name]
                break
        
        # Get sample rows
        sample_df = self.df.head(5)
        sample_data = sample_df.to_dict('records')
        
        return {
            "columns": self.original_columns,
            "row_count": len(self.df),
            "sample_data": sample_data,
            "detected_keyword_column": detected_kw,
            "detected_volume_column": detected_vol,
            "detected_kd_column": detected_kd
        }
    
    def parse_with_selection(
        self,
        keyword_column: str,
        volume_column: Optional[str] = None,
        kd_column: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse using user-selected columns.
        
        Args:
            keyword_column: Name of column containing keywords
            volume_column: Optional column for search volume
            kd_column: Optional column for keyword difficulty
        
        Returns:
            List of keyword dictionaries with metrics
        """
        if self.df is None:
            raise ValidationError("No file loaded. Call preview() first.")
        
        if keyword_column not in self.df.columns:
            raise ValidationError(f"Column '{keyword_column}' not found")
        
        self.keyword_column = keyword_column
        self.volume_column = volume_column
        self.kd_column = kd_column
        
        keywords_data = []
        
        for _, row in self.df.iterrows():
            kw = row[keyword_column]
            if pd.isna(kw):
                continue
            
            kw_str = str(kw).strip()
            if not kw_str:
                continue
            
            entry = {"keyword": kw_str}
            
            if volume_column and volume_column in self.df.columns:
                try:
                    vol = row[volume_column]
                    entry["search_volume"] = int(
                        float(vol)
                    ) if pd.notna(vol) else 0
                except (ValueError, TypeError):
                    entry["search_volume"] = 0
            
            if kd_column and kd_column in self.df.columns:
                try:
                    kd = row[kd_column]
                    entry["keyword_difficulty"] = float(
                        kd
                    ) if pd.notna(kd) else 0
                except (ValueError, TypeError):
                    entry["keyword_difficulty"] = 0
            
            keywords_data.append(entry)
        
        return keywords_data
    
    def parse(
        self,
        file_content: BytesIO,
        encoding: str = "utf-8",
        keyword_column: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse CSV file and extract keywords with optional metrics.
        
        Args:
            file_content: File content as BytesIO
            encoding: File encoding (default: utf-8)
            keyword_column: Specific column name (auto-detect if None)
        
        Returns:
            List of keyword dictionaries with metrics
        
        Raises:
            ValidationError: If parsing fails or no keyword column found
        """
        # Load the file with robust parsing
        try:
            for enc in [encoding, "utf-8", "latin-1", "cp1252"]:
                try:
                    file_content.seek(0)
                    self.df = pd.read_csv(
                        file_content,
                        encoding=enc,
                        low_memory=False,
                        on_bad_lines='skip'
                    )
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValidationError(
                    "Could not decode file with any supported encoding"
                )
        except Exception as e:
            raise ValidationError(f"Failed to parse CSV: {str(e)}")
        
        # Store original columns
        self.original_columns = list(self.df.columns)
        
        # Normalize for detection
        normalized = {
            str(col).lower().strip(): col
            for col in self.df.columns
        }
        
        # Auto-detect keyword column
        detected_kw = None
        for name in self.KEYWORD_COLUMN_NAMES:
            if name in normalized:
                detected_kw = normalized[name]
                break
        
        # Auto-detect volume column
        detected_vol = None
        for name in self.VOLUME_COLUMN_NAMES:
            if name in normalized:
                detected_vol = normalized[name]
                break
        
        # Auto-detect KD column
        detected_kd = None
        for name in self.KD_COLUMN_NAMES:
            if name in normalized:
                detected_kd = normalized[name]
                break
        
        preview = {
            "columns": self.original_columns,
            "row_count": len(self.df),
            "detected_keyword_column": detected_kw,
            "detected_volume_column": detected_vol,
            "detected_kd_column": detected_kd
        }
        
        # Determine keyword column
        if keyword_column:
            kw_col = keyword_column
        elif preview["detected_keyword_column"]:
            kw_col = preview["detected_keyword_column"]
        else:
            raise ValidationError(
                "Could not auto-detect keyword column. "
                f"Available columns: {', '.join(preview['columns'])}"
            )
        
        return self.parse_with_selection(
            keyword_column=kw_col,
            volume_column=preview.get("detected_volume_column"),
            kd_column=preview.get("detected_kd_column")
        )
    
    def _find_column(self, possible_names: List[str]) -> Optional[str]:
        """
        Find a column matching any of the possible names.
        
        Args:
            possible_names: List of possible column name variations
        
        Returns:
            Actual column name if found, None otherwise
        """
        if self.df is None:
            return None
        
        for col in self.df.columns:
            col_clean = col.lower().strip()
            if col_clean in possible_names:
                return col
        
        return None
    
    def get_column_info(self) -> dict:
        """
        Get information about detected columns.
        
        Returns:
            Dict with column detection results
        """
        return {
            "keyword_column": self.keyword_column,
            "volume_column": self.volume_column,
            "kd_column": self.kd_column,
            "has_volume": self.volume_column is not None,
            "has_kd": self.kd_column is not None,
            "total_columns": len(self.df.columns) if self.df is not None else 0,
            "column_names": (
                list(self.df.columns) if self.df is not None else []
            )
        }
    
    def get_sample(self, n: int = 5) -> List[dict]:
        """
        Get sample rows from the parsed data.
        
        Args:
            n: Number of sample rows
        
        Returns:
            List of sample row dictionaries
        """
        if self.df is None or self.keyword_column is None:
            return []
        
        sample_df = self.df.head(n)
        samples = []
        
        for _, row in sample_df.iterrows():
            sample = {"keyword": str(row[self.keyword_column])}
            
            if self.volume_column:
                sample["search_volume"] = row.get(self.volume_column, 0)
            
            if self.kd_column:
                sample["keyword_difficulty"] = row.get(self.kd_column, 0)
            
            samples.append(sample)
        
        return samples


def preview_uploaded_file(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile
) -> Tuple[CSVParser, Dict[str, Any]]:
    """
    Preview a Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Tuple of (parser instance, preview info)
    """
    parser = CSVParser()
    content = BytesIO(uploaded_file.read())
    preview = parser.preview(content)
    return parser, preview


def parse_uploaded_file(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    keyword_column: Optional[str] = None
) -> Tuple[List[Dict], dict]:
    """
    Convenience function to parse Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        keyword_column: Specific column to use for keywords
    
    Returns:
        Tuple of (keywords_data, column_info)
    """
    parser = CSVParser()
    content = BytesIO(uploaded_file.read())
    keywords_data = parser.parse(content, keyword_column=keyword_column)
    column_info = parser.get_column_info()
    return keywords_data, column_info