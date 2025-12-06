"""
CSV export functionality for clustering results.
"""
import csv
import io
from typing import List, Dict, Any
from datetime import datetime


class CSVExporter:
    """Exports clustering results to CSV format."""
    
    def export_clusters(
        self,
        clusters: List[Dict[str, Any]],
        include_keywords: bool = True
    ) -> str:
        """
        Export clusters to CSV string.
        
        Args:
            clusters: List of cluster data dictionaries
            include_keywords: Whether to include keyword details
        
        Returns:
            CSV content as string
        """
        output = io.StringIO()
        
        if include_keywords:
            return self._export_with_keywords(clusters, output)
        else:
            return self._export_summary(clusters, output)
    
    def _export_with_keywords(
        self,
        clusters: List[Dict[str, Any]],
        output: io.StringIO
    ) -> str:
        """Export with one row per keyword."""
        fieldnames = [
            "cluster_id",
            "cluster_label",
            "keyword",
            "search_volume",
            "keyword_difficulty",
            "intent",
            "cluster_total_volume",
            "cluster_avg_difficulty",
            "cluster_size",
            "quality_score"
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for cluster in clusters:
            cluster_id = cluster.get("id", 0)
            label = cluster.get("label", f"Cluster {cluster_id}")
            intent = cluster.get("intent", "unknown")
            total_vol = cluster.get("total_volume", 0)
            avg_diff = cluster.get("avg_difficulty", 0)
            size = cluster.get("size", 0)
            quality = cluster.get("quality_score", 0)
            
            keywords = cluster.get("keywords", [])
            
            for kw_data in keywords:
                if isinstance(kw_data, dict):
                    keyword = kw_data.get("keyword", "")
                    volume = kw_data.get("search_volume", 0)
                    difficulty = kw_data.get("keyword_difficulty", 0)
                else:
                    keyword = str(kw_data)
                    volume = 0
                    difficulty = 0
                
                writer.writerow({
                    "cluster_id": cluster_id,
                    "cluster_label": label,
                    "keyword": keyword,
                    "search_volume": volume,
                    "keyword_difficulty": difficulty,
                    "intent": intent,
                    "cluster_total_volume": total_vol,
                    "cluster_avg_difficulty": round(avg_diff, 2),
                    "cluster_size": size,
                    "quality_score": round(quality, 2)
                })
        
        return output.getvalue()
    
    def _export_summary(
        self,
        clusters: List[Dict[str, Any]],
        output: io.StringIO
    ) -> str:
        """Export with one row per cluster."""
        fieldnames = [
            "cluster_id",
            "label",
            "size",
            "total_volume",
            "avg_difficulty",
            "intent",
            "quality_score",
            "top_keywords"
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for cluster in clusters:
            cluster_id = cluster.get("id", 0)
            keywords = cluster.get("keywords", [])
            
            # Get top 5 keywords
            top_kw = []
            for kw in keywords[:5]:
                if isinstance(kw, dict):
                    top_kw.append(kw.get("keyword", ""))
                else:
                    top_kw.append(str(kw))
            
            writer.writerow({
                "cluster_id": cluster_id,
                "label": cluster.get("label", f"Cluster {cluster_id}"),
                "size": cluster.get("size", len(keywords)),
                "total_volume": cluster.get("total_volume", 0),
                "avg_difficulty": round(
                    cluster.get("avg_difficulty", 0), 2
                ),
                "intent": cluster.get("intent", "unknown"),
                "quality_score": round(
                    cluster.get("quality_score", 0), 2
                ),
                "top_keywords": " | ".join(top_kw)
            })
        
        return output.getvalue()
    
    def export_unclustered(
        self,
        keywords: List[Dict[str, Any]]
    ) -> str:
        """
        Export unclustered keywords.
        
        Args:
            keywords: List of unclustered keyword data
        
        Returns:
            CSV content as string
        """
        output = io.StringIO()
        
        fieldnames = [
            "keyword",
            "search_volume",
            "keyword_difficulty"
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for kw in keywords:
            if isinstance(kw, dict):
                writer.writerow({
                    "keyword": kw.get("keyword", ""),
                    "search_volume": kw.get("search_volume", 0),
                    "keyword_difficulty": kw.get("keyword_difficulty", 0)
                })
            else:
                writer.writerow({
                    "keyword": str(kw),
                    "search_volume": 0,
                    "keyword_difficulty": 0
                })
        
        return output.getvalue()
    
    def export_full_report(
        self,
        job_data: Dict[str, Any]
    ) -> str:
        """
        Export full clustering report.
        
        Args:
            job_data: Complete job data including clusters and stats
        
        Returns:
            CSV content with all data
        """
        output = io.StringIO()
        
        # Write metadata header
        output.write("# Keyword Clustering Report\n")
        output.write(f"# Generated: {datetime.now().isoformat()}\n")
        output.write(f"# Total Keywords: {job_data.get('total_keywords', 0)}\n")
        output.write(f"# Total Clusters: {job_data.get('total_clusters', 0)}\n")
        cr = job_data.get('cluster_rate', 0)
        output.write(f"# Cluster Rate: {cr:.1%}\n")
        output.write("\n")
        
        # Write cluster data
        clusters = job_data.get("clusters", [])
        output.write(self._export_with_keywords(clusters, io.StringIO()))
        
        return output.getvalue()


def create_download_link(
    csv_content: str,
    filename: str
) -> bytes:
    """
    Create downloadable CSV bytes.
    
    Args:
        csv_content: CSV content string
        filename: Suggested filename
    
    Returns:
        UTF-8 encoded bytes with BOM for Excel compatibility
    """
    # Add BOM for Excel UTF-8 compatibility
    bom = b'\xef\xbb\xbf'
    return bom + csv_content.encode('utf-8')