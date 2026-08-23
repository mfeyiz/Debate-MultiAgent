"""Tests for Teyit Lab improvements (URL filtering and query planning)."""

from __future__ import annotations

from app.services.turkish_archive_scraper import TurkishArchiveScraper
from app.services.fact_check_service import FactCheckService


def test_is_valid_source_url_accepts_valid_articles() -> None:
    valid_urls = [
        "https://teyit.org/analiz-izmirdeki-deprem-videosunun-gercek-oldugu-iddiasi",
        "https://teyit.org/makale-dezenformasyonla-mucadele-rehberi",
        "https://teyit.org/kronoloji-secim-sureci",
        "https://www.dogrulukpayi.com/dogrulama/enflasyon-rakamlari-dogru-mu",
        "https://www.dogrulukpayi.com/iddia-kontrolu/chpnin-secim-vaadi-gerceklesmedi-mi",
        "https://www.malumatfurus.org/ataturkun-soyledigi-iddia-edilen-sozler/",
        "https://malumatfurus.org/en-cok-paylasilan-dezenformasyonlar/",
    ]
    for url in valid_urls:
        assert TurkishArchiveScraper.is_valid_source_url(url) is True, f"Failed to accept valid URL: {url}"


def test_is_valid_source_url_rejects_boilerplate_pages() -> None:
    invalid_urls = [
        "https://teyit.org",
        "https://teyit.org/",
        "https://teyit.org/hakkimizda",
        "https://teyit.org/kunye",
        "https://teyit.org/iletisim/",
        "https://teyit.org/kategori/ekonomi/",
        "https://teyit.org/etiket/deprem/",
        "https://teyit.org/yazar/ali-veli/",
        "https://teyit.org/page/3/",
        "https://teyit.org/feed/",
        "https://teyit.org/?s=enflasyon",
        "https://dogrulukpayi.com/kunye",
        "https://dogrulukpayi.com/bulten",
        "https://dogrulukpayi.com/ekip",
        "https://www.malumatfurus.org",
        "https://www.malumatfurus.org/",
        "https://www.malumatfurus.org/malumatfurus-hakkinda/",
        "https://www.malumatfurus.org/iletisim/",
        "https://www.malumatfurus.org/kategori/hatali-bilgiler/",
        "https://www.malumatfurus.org/page/2/?s=ataturk",
    ]
    for url in invalid_urls:
        assert TurkishArchiveScraper.is_valid_source_url(url) is False, f"Failed to reject invalid URL: {url}"


def test_query_plan_for_economic_claim_includes_official_data(monkeypatch) -> None:
    from app.config import Config
    monkeypatch.setattr(Config, "FACT_CHECK_MAX_QUERIES_PER_CLAIM", 3)
    claim_text = "TÜİK yıllık enflasyon oranını yüzde 44.38 olarak açıkladı."
    plan = FactCheckService._query_plan_for_claim(claim_text)
    
    # Check that we query official databases (TÜİK, TCMB)
    scopes = [scope for _, scope in plan]
    assert "official_data" in scopes
    assert "verification" in scopes
    assert "direct_claim" in scopes
    
    # Check that official data query contains appropriate keywords
    official_query = [q for q, scope in plan if scope == "official_data"][0]
    assert "TÜİK" in official_query or "TCMB" in official_query or "Merkez Bankası" in official_query


def test_query_plan_for_general_claim_excludes_official_data(monkeypatch) -> None:
    from app.config import Config
    monkeypatch.setattr(Config, "FACT_CHECK_MAX_QUERIES_PER_CLAIM", 3)
    claim_text = "Sosyal medyada yayılan videonun uzaylı istilasını gösterdiği iddia ediliyor."
    plan = FactCheckService._query_plan_for_claim(claim_text)
    
    scopes = [scope for _, scope in plan]
    assert "official_data" not in scopes
    assert "verification" in scopes
    assert "direct_claim" in scopes
    
    # Check that verification query is targeted
    verify_query = [q for q, scope in plan if scope == "verification"][0]
    assert "teyit" in verify_query or "doğruluk payı" in verify_query or "malumatfuruş" in verify_query
