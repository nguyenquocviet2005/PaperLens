# The UI is hosted on Vercel (project "paperlens"; see src/frontend/vercel.json and
# src/ops/deploy_aws.py ui). paperlens.ailearn-vaic.cloud is this zone's apex, where a
# CNAME isn't allowed, so it uses Vercel's recommended A records.
resource "aws_route53_record" "ui" {
  zone_id = aws_route53_zone.main.zone_id
  name    = var.domain
  type    = "A"
  ttl     = 300
  records = var.vercel_ipv4
}
