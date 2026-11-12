# Transactional email (Supabase Auth confirmations and password resets) is sent through
# Resend from noreply@{var.domain}. Record values come from the Resend domain
# (id 33f778a8-0844-45b6-94a1-1fa793eecdbb, region ap-northeast-1) and are public by nature.

resource "aws_route53_record" "resend_dkim" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "resend._domainkey.${var.domain}"
  type    = "TXT"
  ttl     = 300
  records = ["p=MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQC9z4nlp7/JxvJWIYT6uyxGrgIvcTIaowDh8XpPuz91ohx/eW4PLkgG6vs3x36YXjP0a/9qZC2bTzRYvYrrbRvxU+r7Cqps7Dk1FkQctCBPlAct8wjUiuLhmxgjDHVWzrQZ+XRHT6A7oDahJctnZJahzHQ8vlZN7IZvu0M0lq1NaQIDAQAB"]
}

resource "aws_route53_record" "resend_bounce_mx" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "send.${var.domain}"
  type    = "MX"
  ttl     = 300
  records = ["10 feedback-smtp.ap-northeast-1.amazonses.com"]
}

resource "aws_route53_record" "resend_spf" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "send.${var.domain}"
  type    = "TXT"
  ttl     = 300
  records = ["v=spf1 include:amazonses.com ~all"]
}

resource "aws_route53_record" "resend_return_path" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "rsend.${var.domain}"
  type    = "CNAME"
  ttl     = 300
  records = ["send.forge.rmta.net"]
}

# Monitoring-only DMARC policy; tighten to quarantine once delivery is confirmed.
resource "aws_route53_record" "dmarc" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "_dmarc.${var.domain}"
  type    = "TXT"
  ttl     = 300
  records = ["v=DMARC1; p=none;"]
}
